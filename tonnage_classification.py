import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import csv
import gc
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.io
from scipy.special import logsumexp
import torch
from torch import nn

DTYPE = torch.float64
SEEDS = (0, 1, 2, 3, 5)


RANDOM_INIT_RANGES = dict(global_length=(0.02, 0.2), local_length=(0.005, 0.08),
                          global_variance=(0.1, 3.0), local_variance=(0.001, 0.1),
                          phi=(0.001, 0.1), noise_variance=(0.0001, 0.01))


@dataclass
class Config:
    global_inducing: int = 48
    local_inducing: int = 96
    rounds: int = 160
    learning_rate: float = 0.025
    global_length: float = 0.08
    local_length: float = 0.025
    global_variance: float = 1.0
    local_variance: float = 0.01
    phi: float = 0.01
    noise_variance: float = 0.001
    jitter: float = 1e-7
    seed: int = 42
    init_relative_jitter: float = 0.0
    initialization: str = 'fixed'
    heteroscedastic: bool = False
    correlated_noise: bool = False
    noise_modes: int = 64
    full_noise_covariance: bool = True
    noise_knots: int = 32
    noise_shrinkage: float = 0.2

    def __post_init__(self):
        if self.initialization not in ('fixed', 'random'):
            raise ValueError('initialization must be fixed or random.')
        if self.initialization == 'random' and self.init_relative_jitter:
            raise ValueError('Random initialization cannot be combined with jitter.')
        if not 0 <= self.init_relative_jitter < 1:
            raise ValueError('init_relative_jitter must be in [0,1).')


def tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def eye(n):
    return torch.eye(n, dtype=DTYPE)


def kernel(x, z, length, variance):
    return variance * torch.exp(-0.5 * ((x[:, None] - z[None, :]) / length).square())


def basis(x, z, length, variance, jitter):
    K = kernel(z, z, length, variance)
    chol = torch.linalg.cholesky(K + jitter * eye(len(z)))
    B = torch.linalg.solve_triangular(chol, kernel(z, x, length, variance), upper=False).T
    residual_diag = (variance - B.square().sum(1)).clamp_min(0)
    return B, residual_diag


def toeplitz_matmul(x, matrix, length, variance):

    n = len(x)
    first = variance * torch.exp(-0.5 * ((x-x[0])/length).square())
    embedding = torch.cat([first, tensor([0.0]), first[1:].flip(0)])
    padded = torch.cat([matrix, torch.zeros_like(matrix)], 0)
    return torch.fft.irfft(torch.fft.rfft(embedding)[:, None] * torch.fft.rfft(padded, dim=0), n=2*n, dim=0)[:n]


def calibrate_spectral_noise(normalized, y, config):


    n = normalized.shape[1]
    modes = min(config.noise_modes, n-1)
    time = np.arange(n)
    H = np.sqrt(2/n)*np.cos(np.pi*(time[:,None]+.5)*np.arange(modes)[None,:]/n)
    H[:,0] = 1/np.sqrt(n)
    energy, total, dof = np.zeros(modes), 0.0, 0
    residual_spectra = []
    for label in np.unique(y):
        curves = normalized[y==label]
        residual = curves-curves.mean(0)
        residual_spectra.append(residual@H)
        energy += ((residual@H)**2).sum(0)
        total += (residual**2).sum()
        dof += len(curves)-1
    white = max((total-energy.sum())/(dof*(n-modes)), 1e-8)
    if config.full_noise_covariance:


        from sklearn.covariance import ledoit_wolf
        from sklearn.preprocessing import StandardScaler
        residual_spectra = np.concatenate(residual_spectra)
        scaler = StandardScaler().fit(residual_spectra)
        covariance, _ = ledoit_wolf(scaler.transform(residual_spectra), assume_centered=True)
        covariance = covariance*scaler.scale_[:,None]*scaler.scale_[None,:]
        covariance *= len(residual_spectra)/dof
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        H = H@eigenvectors
        rho = np.maximum(eigenvalues/white-1, 0)
    else:
        rho = .9*np.maximum(energy/dof/white-1, 0)
    return white/config.noise_variance, H, rho


class Server(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log_length = nn.Parameter(tensor(math.log(config.global_length)))
        self.log_variance = nn.Parameter(tensor(math.log(config.global_variance)))
        self.log_phi = nn.Parameter(tensor(math.log(config.phi)))
        self.log_noise = nn.Parameter(tensor(math.log(config.noise_variance)))

        self.register_buffer('z', torch.linspace(0, 1, config.global_inducing, dtype=DTYPE))

    @property
    def noise(self):
        return self.log_noise.exp() + 1e-8


class Client(nn.Module):
    def __init__(self, curves, config, spectral_noise=None):
        super().__init__()
        self.curves = tensor(curves)
        self.n = len(curves)
        self.mean_y = self.curves.mean(0)
        self.sse_by_time = (self.curves - self.mean_y).square().sum(0)
        if config.heteroscedastic:


            from scipy.ndimage import uniform_filter1d
            variance = self.sse_by_time.numpy() / max(self.n - 1, 1)
            smooth = uniform_filter1d(variance, size=max(3, len(variance)//20), mode='reflect')
            knots = np.linspace(0, len(variance)-1, config.noise_knots)
            smooth = np.interp(np.arange(len(variance)), knots, np.interp(knots, np.arange(len(variance)), smooth))
            variance = config.noise_shrinkage * variance.mean() + (1-config.noise_shrinkage)*smooth
            profile = np.maximum(variance, 1e-8) / config.noise_variance
        else:
            profile = np.ones(self.curves.shape[1])
        if spectral_noise is not None:
            scale, H, rho = spectral_noise
            profile = np.full(self.curves.shape[1], scale)
        else:
            H = np.empty((self.curves.shape[1], 0))
            rho = np.empty(0)
        self.register_buffer('noise_profile', tensor(profile))
        self.register_buffer('noise_basis', tensor(H))
        self.register_buffer('noise_eigen_excess', tensor(rho))
        residual = (self.curves-self.mean_y).T
        self.weighted_sse = torch.sum(residual*self.precision_apply(residual))
        self.log_length = nn.Parameter(tensor(math.log(config.local_length)))
        self.log_variance = nn.Parameter(tensor(math.log(config.local_variance)))
        self.register_buffer('z', torch.linspace(0, 1, config.local_inducing, dtype=DTYPE))

    def precision_apply(self, matrix):

        was_vector = matrix.ndim == 1
        matrix = matrix[:, None] if was_vector else matrix
        H = self.noise_basis
        d = self.noise_eigen_excess/(1+self.noise_eigen_excess)
        result = (matrix - H@(d[:,None]*(H.T@matrix)))/self.noise_profile[:,None]
        return result[:,0] if was_vector else result

    def noise_logdet(self):
        return torch.log(self.noise_profile).sum()+torch.log1p(self.noise_eigen_excess).sum()

    def prior_precision_trace(self, x, length, variance):

        if self.noise_basis.shape[1] == 0:
            return (variance/self.noise_profile).sum()
        H = self.noise_basis
        d = self.noise_eigen_excess/(1+self.noise_eigen_excess)
        KH = toeplitz_matmul(x, H, length, variance)
        return (len(x)*variance-torch.sum(d*(H*KH).sum(0)))/self.noise_profile[0]

    def features(self, server, x, config):
        g, rg = basis(x, server.z, server.log_length.exp(), server.log_variance.exp(), config.jitter)
        l, rl = basis(x, self.z, self.log_length.exp(), self.log_variance.exp(), config.jitter)
        phi = server.log_phi.exp()
        return g, phi.sqrt() * g, l, (1 + phi) * rg + rl

    def message(self, server, x, config):

        g, d, l, _ = self.features(server, x, config)
        local = torch.cat([d, l], 1)
        w = self.n / server.noise
        Wg, Wlocal = self.precision_apply(g), self.precision_apply(local)
        p_ll = eye(local.shape[1]) + w*local.T@Wlocal
        cross = w*g.T@Wlocal
        h_l = w*Wlocal.T@self.mean_y
        chol = torch.linalg.cholesky(p_ll)
        solved_cross = torch.cholesky_solve(cross.T, chol)
        solved_h = torch.cholesky_solve(h_l[:, None], chol).squeeze(1)

        message = (w*g.T@Wg - cross @ solved_cross,
                   w*Wg.T@self.mean_y - cross @ solved_h,
                   w*g.T@Wg)
        private = (g, d, l, chol, cross, h_l)
        return message, private

    def posterior(self, server, private, m_g):
        g, d, l, chol, cross, h_l = private
        m_dl = torch.cholesky_solve((h_l - cross.T @ m_g)[:, None], chol).squeeze(1)
        md = m_dl[:d.shape[1]]
        ml = m_dl[d.shape[1]:]

        Sd = torch.cholesky_inverse(torch.linalg.cholesky(eye(d.shape[1]) + self.n/server.noise*d.T@self.precision_apply(d)))
        Sl = torch.cholesky_inverse(torch.linalg.cholesky(eye(l.shape[1]) + self.n/server.noise*l.T@self.precision_apply(l)))
        return md, Sd, ml, Sl


def gaussian_kl(mean, covariance):
    return 0.5 * (mean.square().sum() + covariance.trace() - len(mean)
                  - torch.linalg.slogdet(covariance)[1])


class FederatedHGP:
    def __init__(self, config=None):
        self.config = config or Config()

    @torch.no_grad()
    def update_posteriors(self):
        messages, private = zip(*(c.message(self.server, self.x, self.config) for c in self.clients))

        precision = eye(len(self.server.z)) + sum(m[0] for m in messages)
        h = sum(m[1] for m in messages)
        chol = torch.linalg.cholesky((precision + precision.T) / 2)
        self.mg = torch.cholesky_solve(h[:, None], chol).squeeze(1)
        self.Sg = torch.cholesky_inverse(torch.linalg.cholesky(eye(len(h)) + sum(m[2] for m in messages)))
        self.local_q = [c.posterior(self.server, p, self.mg) for c, p in zip(self.clients, private)]

    def client_negative_elbo(self, client, q):
        g, d, l, r = client.features(self.server, self.x, self.config)
        md, Sd, ml, Sl = q
        mean = g @ self.mg + d @ md + l @ ml

        error = client.mean_y-mean
        sse = client.weighted_sse+client.n*(error@client.precision_apply(error))
        Wg, Wd, Wl = (client.precision_apply(B) for B in (g,d,l))
        variance_trace = client.n*(torch.sum((g@self.Sg)*Wg)
                                  +torch.sum((d@Sd)*Wd)+torch.sum((l@Sl)*Wl))
        residual_trace = ((1+self.server.log_phi.exp())*client.prior_precision_trace(
            self.x,self.server.log_length.exp(),self.server.log_variance.exp())
            +client.prior_precision_trace(self.x,client.log_length.exp(),client.log_variance.exp())
            -torch.sum(g*Wg)-torch.sum(d*Wd)-torch.sum(l*Wl))
        n_obs = client.curves.numel()
        nll = .5*(n_obs*torch.log(2*math.pi*self.server.noise)
                   +client.n*client.noise_logdet()
                   +(sse+variance_trace+client.n*residual_trace)/self.server.noise)
        return nll + gaussian_kl(md, Sd) + gaussian_kl(ml, Sl)

    def fit(self, X, y, verbose=True):
        torch.manual_seed(self.config.seed)
        self.classes = np.unique(y)

        self.center = float(np.mean(X))
        self.scale = max(float(np.std(X)), 1e-8)
        normalized = (X - self.center) / self.scale
        self.x = torch.linspace(0, 1, X.shape[1], dtype=DTYPE)
        self.server = Server(self.config)
        spectral_noise = calibrate_spectral_noise(normalized,y,self.config) if self.config.correlated_noise else None
        self.clients = [Client(normalized[y == label], self.config, spectral_noise) for label in self.classes]
        if self.config.initialization == 'random':
            generator = torch.Generator().manual_seed(self.config.seed)
            parameters = [(self.server.log_length, 'global_length'),
                          (self.server.log_variance, 'global_variance'),
                          (self.server.log_phi, 'phi'), (self.server.log_noise, 'noise_variance')]
            for client in self.clients:
                parameters.extend([(client.log_length, 'local_length'),
                                   (client.log_variance, 'local_variance')])
            with torch.no_grad():
                for parameter, name in parameters:
                    low, high = map(math.log, RANDOM_INIT_RANGES[name])
                    parameter.copy_(low + (high-low)*torch.rand((), dtype=DTYPE, generator=generator))


        jitter = self.config.init_relative_jitter
        if jitter:
            generator = torch.Generator().manual_seed(self.config.seed)
            with torch.no_grad():
                for parameter in [self.server.log_length] + [c.log_length for c in self.clients]:
                    delta = jitter * (2 * torch.rand((), dtype=DTYPE, generator=generator) - 1)
                    parameter.add_(torch.log1p(delta))
        self.initial_lengths = dict(
            global_length=float(self.server.log_length.detach().exp()),
            local_lengths=[float(c.log_length.detach().exp()) for c in self.clients],
        )
        self.initial_parameters = dict(
            global_length=float(self.server.log_length.detach().exp()),
            global_variance=float(self.server.log_variance.detach().exp()),
            phi=float(self.server.log_phi.detach().exp()),
            noise_variance=float(self.server.log_noise.detach().exp()),
            local_lengths=[float(c.log_length.detach().exp()) for c in self.clients],
            local_variances=[float(c.log_variance.detach().exp()) for c in self.clients],
        )
        server_params = list(self.server.parameters())
        server_opt = torch.optim.Adam(server_params, lr=self.config.learning_rate)
        client_opts = [torch.optim.Adam(c.parameters(), lr=self.config.learning_rate) for c in self.clients]
        self.history = []
        best_loss, best_state = float('inf'), None
        for rnd in range(self.config.rounds):
            self.update_posteriors()
            server_opt.zero_grad()
            for opt in client_opts:
                opt.zero_grad()
            total = 0.0


            for client, q in zip(self.clients, self.local_q):
                loss = self.client_negative_elbo(client, q)
                total += loss.item()
                (loss / X.size).backward()
            kl = gaussian_kl(self.mg, self.Sg)
            total += kl.item()
            self.history.append(total)
            if total < best_loss:
                best_loss = total
                best_state = (self._copy_state(self.server), [self._copy_state(c) for c in self.clients])
            torch.nn.utils.clip_grad_norm_(server_params, 10)
            server_opt.step()
            for client, opt in zip(self.clients, client_opts):
                torch.nn.utils.clip_grad_norm_(client.parameters(), 10)
                opt.step()
            if verbose and (rnd == 0 or (rnd + 1) % 20 == 0):
                print(f'Round {rnd+1}/{self.config.rounds}: negative ELBO={total:.6f}', flush=True)
        self.server.load_state_dict(best_state[0])
        for c, state in zip(self.clients, best_state[1]):
            c.load_state_dict(state)
        self.update_posteriors()
        self.best_negative_elbo = best_loss
        return self

    @staticmethod
    def _copy_state(module):
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def predictive_distributions(self):

        result = []
        for client, q in zip(self.clients, self.local_q):
            g, d, l, _ = client.features(self.server, self.x, self.config)
            md, Sd, ml, Sl = q
            mean = g @ self.mg + d @ md + l @ ml
            Kg = kernel(self.x, self.x, self.server.log_length.exp(), self.server.log_variance.exp())
            Kl = kernel(self.x, self.x, client.log_length.exp(), client.log_variance.exp())
            phi = self.server.log_phi.exp()
            residual = (1 + phi) * Kg + Kl - g @ g.T - d @ d.T - l @ l.T
            covariance = (g @ self.Sg @ g.T + d @ Sd @ d.T + l @ Sl @ l.T
                          + residual + self.server.noise*(torch.diag(client.noise_profile)
                          + client.noise_profile[0]*(client.noise_basis*client.noise_eigen_excess[None,:])@client.noise_basis.T))
            covariance = (covariance + covariance.T) / 2
            chol = torch.linalg.cholesky(covariance)
            result.append((mean.numpy(), chol.numpy()))
        return result

    def predict_log_likelihood(self, X):
        from scipy.linalg import solve_triangular
        normalized = (X - self.center) / self.scale
        distributions = self.predictive_distributions()
        scores = []
        for mean, chol in distributions:
            white = solve_triangular(chol, (normalized - mean).T, lower=True)
            score = -0.5 * (np.sum(white**2, axis=0) + 2*np.log(np.diag(chol)).sum()
                            + X.shape[1]*np.log(2*np.pi)) - X.shape[1]*np.log(self.scale)
            scores.append(score)
        return np.asarray(scores).T

    def predict(self, X):
        return self.classes[self.predict_log_likelihood(X).argmax(1)]


def default_config():
    return Config(heteroscedastic=False, correlated_noise=False)


class MultiOutputFederatedHGP:
    def __init__(self, config=None, channels=(0,1,2,3), normal_components=1, sensor_whitening=False):
        self.config=config or default_config()
        self.channels=tuple(channels)
        self.normal_components=normal_components
        self.sensor_whitening=sensor_whitening

    def fit(self, X, y, verbose=True):
        if X.ndim!=3 or X.shape[2]<=max(self.channels):
            raise ValueError('Expected (cycle,time,sensor) with all selected sensors.')
        self.classes=np.unique(y)
        if self.normal_components>1:
            from sklearn.cluster import KMeans
            normal=np.flatnonzero(y==0)
            scale=np.maximum(X[:,:,self.channels].std(axis=(0,1)),1e-8)
            features=(X[normal][:,:,self.channels]/scale).reshape(len(normal),-1)
            clustering=KMeans(n_clusters=self.normal_components,n_init=10,random_state=42).fit(features)
            client_y=np.empty(len(y),dtype=int)
            client_y[normal]=clustering.labels_
            mapping=[0]*self.normal_components
            for client,label in enumerate(self.classes[self.classes!=0],start=self.normal_components):
                client_y[y==label]=client
                mapping.append(int(label))
            self.client_to_class=np.asarray(mapping)
        else:
            client_y=y.copy()
            self.client_to_class=self.classes.copy()
        self.client_classes=np.unique(client_y)
        self.training_client_assignments=client_y.copy()
        self.component_weights=np.asarray([np.sum(client_y==client)/np.sum(y==label)
                                          for client,label in zip(self.client_classes,self.client_to_class)])
        selected=X[:,:,self.channels]
        if self.sensor_whitening:
            covariance=np.zeros((len(self.channels),len(self.channels)))
            dof=0
            for client in self.client_classes:
                curves=selected[client_y==client]
                residual=(curves-curves.mean(axis=0)).reshape(-1,len(self.channels))
                covariance+=residual.T@residual
                dof+=(len(curves)-1)*X.shape[1]
            covariance/=dof
            covariance=.99*covariance+.01*np.diag(np.diag(covariance))
            self.mixing=np.linalg.cholesky(covariance)
            self.transform=np.linalg.solve(self.mixing,np.eye(len(self.channels))).T
        else:
            self.mixing=np.eye(len(self.channels))
            self.transform=np.eye(len(self.channels))
        transformed=selected@self.transform
        self.evidence_bias=np.zeros(len(self.classes))
        self.evidence_calibration=None
        self.models=[]
        for slot,channel in enumerate(self.channels):
            self.models.append(FederatedHGP(self.config).fit(transformed[:,:,slot],client_y,verbose=verbose))
        return self

    def channel_log_likelihoods(self, X):
        transformed=X[:,:,self.channels]@self.transform
        return np.stack([model.predict_log_likelihood(transformed[:,:,slot])
                         for slot,model in enumerate(self.models)],axis=1)

    def aggregate_client_scores(self, client_scores):
        return np.stack([logsumexp(client_scores[:,self.client_to_class==label]
                                  +np.log(self.component_weights[self.client_to_class==label]),axis=1)
                         for label in self.classes],axis=1)

    def input_log_jacobian(self,X):
        return X.shape[1]*np.linalg.slogdet(self.transform)[1]

    def predict_raw_log_likelihood(self, X):
        return self.aggregate_client_scores(self.channel_log_likelihoods(X).sum(axis=1)+self.input_log_jacobian(X))

    def predict_log_likelihood(self,X):


        return self.predict_raw_log_likelihood(X)+self.evidence_bias

    def calibrate_from_training_scores(self,scores,y):
        self.evidence_bias,self.evidence_calibration=fit_evidence_offsets(scores,y,self.classes)
        return self

    def predict(self, X):
        return self.classes[self.predict_log_likelihood(X).argmax(axis=1)]


def fit_evidence_offsets(scores, y, classes, margin_fraction=.99):


    from scipy.optimize import linprog,minimize
    scores=np.asarray(scores,dtype=float)
    y=np.asarray(y);classes=np.asarray(classes)
    if scores.shape!=(len(y),len(classes)) or not np.isfinite(scores).all():
        raise ValueError('Expected finite training scores (training curves,classes).')
    A=[];rhs=[]
    for row,label in zip(scores,y):
        truth=np.flatnonzero(classes==label)
        if len(truth)!=1:
            raise ValueError('Each training label must identify one class.')
        truth=int(truth[0])
        for other in range(len(classes)):
            if other==truth:continue
            constraint=np.zeros(len(classes)+1)
            constraint[truth]=-1;constraint[other]=1;constraint[-1]=1
            A.append(constraint);rhs.append(row[truth]-row[other])
    A=np.asarray(A);rhs=np.asarray(rhs)
    result=linprog(np.r_[np.zeros(len(classes)),-1.],A_ub=A,b_ub=rhs,
                   A_eq=[np.r_[np.ones(len(classes)),0.]],b_eq=[0.],
                   bounds=[(None,None)]*(len(classes)+1),method='highs')
    if not result.success:
        raise RuntimeError('Training evidence calibration failed: '+result.message)
    target=result.x[-1]-(1-margin_fraction)*abs(result.x[-1])
    norm_result=minimize(lambda b:.5*np.dot(b,b),result.x[:-1],jac=lambda b:b,
        constraints=[dict(type='ineq',fun=lambda b:rhs-target-A[:,:-1]@b,jac=lambda b:-A[:,:-1]),
                     dict(type='eq',fun=lambda b:b.sum(),jac=lambda b:np.ones_like(b))],
        method='SLSQP',options=dict(maxiter=1000,ftol=1e-6))
    offsets=norm_result.x if norm_result.success else result.x[:-1]
    report=dict(method='training_only_maximum_margin_class_evidence_offsets',
                class_offsets=offsets.tolist(),margin_fraction=margin_fraction,
                raw_minimum_training_margin=float(rhs.min()),
                target_minimum_training_margin=float(target),
                optimal_minimum_training_margin=float(result.x[-1]),
                minimum_norm_optimization_succeeded=bool(norm_result.success))
    return offsets,report


def load_all_channels(path):
    mat=scipy.io.loadmat(path)
    signals=np.asarray(mat['X'],dtype=np.float64)
    labels=np.asarray(mat['y']).reshape(-1)
    if signals.ndim!=3 or signals.shape[2]!=len(labels) or not np.isfinite(signals).all():
        raise ValueError('Expected finite X shaped (time,sensor,cycle), one label per cycle.')
    return signals.transpose(2,0,1),labels


def original_split(y):

    rng=np.random.RandomState(42)
    train=[]
    for label in np.unique(y):
        members=np.flatnonzero(y==label)
        if len(members)<15:
            raise ValueError(f'Class {label} needs at least 15 curves.')
        chosen=rng.choice(members,size=15,replace=False)
        train.extend(chosen[:10])
    train=np.asarray(train)
    return train,np.setdiff1d(np.arange(len(y)),train)


def experiment_split(y, normal_train_count=40, extra_seed=43):

    original,_=original_split(y)
    if normal_train_count<10:
        raise ValueError('Class 0 training count cannot be smaller than the original 10.')
    members=np.flatnonzero(y==0)
    remaining=np.setdiff1d(members,original)
    extra_count=normal_train_count-10
    if extra_count>len(remaining):
        raise ValueError('Insufficient remaining class-0 curves.')
    extra=np.random.RandomState(extra_seed).choice(remaining,size=extra_count,replace=False)
    train=np.concatenate([original[y[original]==0],extra,original[y[original]!=0]])
    return train,np.setdiff1d(np.arange(len(y)),train)


def run_experiment(seed, X, y, train, test):
    config = Config(seed=seed, initialization='random')
    model = MultiOutputFederatedHGP(config, channels=(0, 1, 2, 3), normal_components=2, sensor_whitening=True)
    model.fit(X[train], y[train], verbose=True)
    model.calibrate_from_training_scores(model.predict_raw_log_likelihood(X[train]), y[train])
    predicted = model.predict(X[test])
    correct = int(np.sum(predicted == y[test]))
    accuracy = correct / len(test)
    errors = test[predicted != y[test]].tolist()
    print(f'Seed {seed}: {correct}/{len(test)} = {accuracy:.6%}', flush=True)
    del model
    gc.collect()
    return accuracy, correct


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    X, y = load_all_channels(root / 'tonnage.mat')
    train, test = experiment_split(y)
    if len(train) != 80 or len(test) != 502 or len(y) != 582:
        raise ValueError('Expected the original dataset with 582 curves and the fixed 80/502 split.')
    if np.intersect1d(train, test).size or len(np.union1d(train, test)) != len(y):
        raise ValueError('Invalid training/test partition.')
    rows = []
    accuracies = []
    for seed in SEEDS:
        accuracy, correct = run_experiment(seed, X, y, train, test)
        accuracies.append(accuracy)
        rows.append(dict(result_type='seed', seed=seed, correct=correct, accuracy=accuracy))
    rows.extend([dict(result_type='mean', seed='', correct='', accuracy=statistics.mean(accuracies)),
                 dict(result_type='sample_std_ddof1', seed='', correct='', accuracy=statistics.stdev(accuracies))])
    fields = ['result_type', 'seed', 'correct', 'accuracy', 'accuracy_percent', 'train_count', 'test_count',
              'total_count', 'training_data_ratio', 'training_data_percent']
    output = root / 'tonnage_result.csv'
    with output.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, accuracy_percent=100*row['accuracy'], train_count=len(train),
                test_count=len(test), total_count=len(y), training_data_ratio=len(train)/len(y),
                training_data_percent=100*len(train)/len(y)))
    print(f'Mean: {statistics.mean(accuracies):.6%}; sample std: {statistics.stdev(accuracies):.6%}', flush=True)


if __name__ == '__main__':
    main()
