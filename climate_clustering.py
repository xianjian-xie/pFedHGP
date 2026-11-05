import argparse
import json
from pathlib import Path

import glob
import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.linalg import block_diag
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ALL_STATIONS = [
    "Gucheng", "Huairou", "Tiantan", "Changping", "Guanyuan",
    "Nongzhanguan", "Wanliu", "Dongsi", "Wanshouxigong",
    "Aotizhongxin", "Dingling", "Shunyi"
]

URBAN = {"Gucheng", "Tiantan", "Guanyuan", "Nongzhanguan",
         "Wanliu", "Dongsi", "Wanshouxigong", "Aotizhongxin"}
WEST_SUBURB = {"Changping", "Dingling"}
EAST_SUBURB = {"Huairou", "Shunyi"}

TRUE_3CLASS = {s: (0 if s in URBAN else (1 if s in WEST_SUBURB else 2)) for s in ALL_STATIONS}
CLASS_NAMES = {0: "Urban", 1: "West-Suburb", 2: "East-Suburb"}
CMAP3 = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

STATION_COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping":    (40.217, 116.231),
    "Dingling":     (40.289, 116.233),
    "Dongsi":       (39.929, 116.417),
    "Guanyuan":     (39.929, 116.339),
    "Gucheng":      (39.914, 116.184),
    "Huairou":      (40.357, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi":       (40.127, 116.655),
    "Tiantan":      (39.874, 116.417),
    "Wanliu":       (39.999, 116.307),
    "Wanshouxigong":(39.878, 116.352),
}

LABEL_OFFSETS = {
    "Aotizhongxin": (0.012, 0.018, "left", "bottom"),
    "Changping": (-0.018, 0.010, "right", "bottom"),
    "Dingling": (-0.028, 0.002, "right", "bottom"),
    "Dongsi": (-0.037, -0.040, "left", "bottom"),
    "Guanyuan": (-0.014, 0.015, "right", "center"),
    "Gucheng": (0.024, -0.014, "right", "top"),
    "Huairou": (0.014, 0.012, "left", "bottom"),
    "Nongzhanguan": (0.012, -0.008, "left", "top"),
    "Shunyi": (0.014, -0.010, "left", "top"),
    "Tiantan": (0.012, -0.014, "left", "top"),
    "Wanliu": (-0.014, 0.014, "right", "bottom"),
    "Wanshouxigong": (-0.014, -0.014, "right", "top"),
}

FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN"]


def load_beijing_local(pattern="./beijing/PRSA_Data_*.csv", station_whitelist=ALL_STATIONS):
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No file matched: {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, encoding="gbk")

        if "station" in df.columns:
            st = str(df["station"].iloc[0])
        else:
            base = os.path.basename(f)
            parts = base.replace(".csv", "").split("_")
            st = parts[-2] if len(parts) >= 2 else base
            df["station"] = st

        if st not in station_whitelist:
            continue

        if all(c in df.columns for c in ["year", "month", "day", "hour"]):
            df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        elif "Date" in df.columns:
            df["datetime"] = pd.to_datetime(df["Date"])
            df["year"] = df["datetime"].dt.year
            df["month"] = df["datetime"].dt.month
            df["day"] = df["datetime"].dt.day
            df["hour"] = df["datetime"].dt.hour
        else:
            raise RuntimeError(f"Cannot build datetime for file: {f}")

        keep = ["datetime", "station", "year", "month", "day", "hour"] + [
            c for c in FEATURES if c in df.columns
        ]
        df = df[keep].dropna(subset=["datetime", "station"])
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data[data["station"].isin(station_whitelist)].copy()
    data = data.sort_values(["station", "datetime"]).reset_index(drop=True)

    for ft in FEATURES:
        if ft not in data.columns:
            data[ft] = np.nan

    return data


def rbf_kernel(x, y, ell, sig2):
    x = np.atleast_2d(x).astype(float)
    y = np.atleast_2d(y).astype(float)
    d2 = (x - y.T) ** 2
    return sig2 * np.exp(-0.5 * d2 / (ell**2 + 1e-12))


def projection_operator(times_len, kernel_fn, Z, noise_sigma2=1e-2):
    t = np.arange(times_len, dtype=float).reshape(-1, 1)
    Z = np.asarray(Z, dtype=float).reshape(-1, 1)
    Kzx = kernel_fn(Z, t)
    Kxx = kernel_fn(t, t)
    lam = noise_sigma2 * (float(np.median(np.diag(Kxx))) if Kxx.size else 1.0)
    A = Kzx @ np.linalg.solve(Kxx + lam * np.eye(Kxx.shape[0]), Kzx.T)
    return A, Kzx, Kxx, lam


class FederatedGPClient(gpytorch.models.ApproximateGP):
    def __init__(self, Z_g, Z_i, ell_g=50.0, sig2_g=1.0, ell_i=10.0, sig2_i=0.5):
        M, _ = Z_g.shape
        M_i, _ = Z_i.shape
        dummy = torch.cat([Z_g, Z_i], dim=0)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(dummy.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, dummy, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.inducing_points = torch.nn.Parameter(dummy.clone())
        self.variational_strategy.inducing_points = self.inducing_points

        self.mean_module = gpytorch.means.ZeroMean()

        self.lengthscale_g = torch.nn.Parameter(torch.tensor(float(ell_g)))
        self.sigma2_g      = torch.nn.Parameter(torch.tensor(float(sig2_g)))
        self.lengthscale_i = torch.nn.Parameter(torch.tensor(float(ell_i)))
        self.sigma2_i      = torch.nn.Parameter(torch.tensor(float(sig2_i)))

        self.u_g     = torch.nn.Parameter(torch.randn(M))
        self.delta   = torch.nn.Parameter(torch.randn(M))
        self.u_local = torch.nn.Parameter(torch.randn(M_i))
        self.num_global, self.num_local = M, M_i

    @property
    def Z_g(self):
        return self.inducing_points[:self.num_global]

    @property
    def Z_i(self):
        return self.inducing_points[self.num_global:]

    def components(self, x):
        jitter = 1e-3
        Zg, Zi = self.Z_g, self.Z_i

        diff_g = x.unsqueeze(1) - Zg.unsqueeze(0)
        Kgx = self.sigma2_g * torch.exp(-0.5 * (diff_g**2).squeeze(-1) / self.lengthscale_g**2)
        KgZ = self.sigma2_g * torch.exp(-0.5 * ((Zg.unsqueeze(1)-Zg.unsqueeze(0))**2).squeeze(-1) / self.lengthscale_g**2)
        Kg_inv = torch.inverse(KgZ + jitter * torch.eye(Zg.size(0)))
        f_global = Kgx @ (Kg_inv @ (self.u_g + self.delta))

        diff_i = x.unsqueeze(1) - Zi.unsqueeze(0)
        Kix = self.sigma2_i * torch.exp(-0.5 * (diff_i**2).squeeze(-1) / self.lengthscale_i**2)
        KiZ = self.sigma2_i * torch.exp(-0.5 * ((Zi.unsqueeze(1)-Zi.unsqueeze(0))**2).squeeze(-1) / self.lengthscale_i**2)
        Ki_inv = torch.inverse(KiZ + jitter * torch.eye(Zi.size(0)))
        f_local = Kix @ (Ki_inv @ self.u_local)
        return f_global, f_local

    def forward(self, x):
        f_global, f_local = self.components(x)
        mean = f_global + f_local
        cov = torch.eye(x.size(0)) * 0.1
        return gpytorch.distributions.MultivariateNormal(mean, cov)


def station_series(d_station, feature):
    d = d_station.dropna(subset=[feature]).sort_values("datetime").copy()
    d = d.groupby(["date_only", "hour"], as_index=False).first().sort_values(["date_only", "hour"])
    return d[feature].astype(float).values


def build_clients(df_year, feature):
    series = {s: station_series(d, feature) for s, d in df_year.groupby("station")}
    series = {s: y for s, y in series.items() if len(y) >= 8}
    ally = np.concatenate(list(series.values()))
    ym, ys = ally.mean(), ally.std()
    clients = {}
    for s, y in series.items():
        x = torch.arange(len(y), dtype=torch.float32).reshape(-1, 1)
        clients[s] = {'x': x, 'y': torch.from_numpy((y - ym) / ys).float()}
    return clients, ym, ys


def federated_train(clients_data, p, num_rounds=10, local_iters=100, lr=0.1):
    models, liks = [], []
    for data in clients_data:
        L = data['x'].size(0)
        Z_g = torch.linspace(0, L - 1, min(p['M_g'], max(5, L // 2))).unsqueeze(-1)
        Z_i = torch.linspace(0, L - 1, min(p['M_i'], max(4, L // 4))).unsqueeze(-1)
        m = FederatedGPClient(Z_g, Z_i, p['ell_g'], p['sig2_g'], p['ell_i'], p['sig2_i'])
        for q in (m.lengthscale_g, m.sigma2_g, m.lengthscale_i, m.sigma2_i, m.inducing_points):
            q.requires_grad_(False)
        models.append(m)
        liks.append(gpytorch.likelihoods.GaussianLikelihood())

    for rnd in range(1, num_rounds+1):
        print(f"Federated Round {rnd}/{num_rounds}")
        for i, data in enumerate(clients_data):
            m, lk = models[i], liks[i]
            m.train(); lk.train()
            optimizer = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=lr)
            mll = gpytorch.mlls.VariationalELBO(lk, m, num_data=data['x'].size(0))

            for it in range(1, local_iters+1):
                optimizer.zero_grad()
                out = m(data['x'])
                loss = -mll(out, data['y'])
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            agg = sum(m.u_g for m in models) / len(models)
            for m in models:
                m.u_g.copy_(agg)

    for m, lk in zip(models, liks):
        m.eval(); lk.eval()
    return models, liks


@torch.no_grad()
def build_fedhgp_station_embed(m, L, ym, ys, noise_sigma2=1e-2):
    Kg = lambda x, y: rbf_kernel(x, y, float(m.lengthscale_g), float(m.sigma2_g))
    Ki = lambda x, y: rbf_kernel(x, y, float(m.lengthscale_i), float(m.sigma2_i))
    A_delta = projection_operator(L, Kg, m.Z_g.numpy(), noise_sigma2)[0]
    A_loc = projection_operator(L, Ki, m.Z_i.numpy(), noise_sigma2)[0]

    A = block_diag(A_delta, A_loc)
    Ahat = A / (np.trace(A) + 1e-12)

    m_delta = sum(m.components(m.Z_g)).double().numpy() * ys + ym
    m_loc = sum(m.components(m.Z_i)).double().numpy() * ys + ym
    h = np.concatenate([m_delta, m_loc])
    h = h / (np.linalg.norm(h) + 1e-12)

    return Ahat, h


def build_all_Ahat_h(df_year, p, seed=0):
    torch.manual_seed(seed)
    clients, ym, ys = build_clients(df_year, p["feature"])
    stations = sorted(clients, key=lambda s: s.lower())
    models, _ = federated_train([clients[s] for s in stations], p, p["num_rounds"], p["local_iters"], p["lr"])
    Ahat_dict, h_dict = {}, {}
    for s, m in zip(stations, models):
        Ahat_dict[s], h_dict[s] = build_fedhgp_station_embed(m, clients[s]['x'].size(0), ym, ys, p["noise_sigma2"])
    return Ahat_dict, h_dict


def pairwise_mats(stations, Ahat, hdict, alpha=0.5):
    n = len(stations)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            DF = np.linalg.norm(Ahat[stations[i]] - Ahat[stations[j]], ord="fro") ** 2
            DH = np.linalg.norm(hdict[stations[i]] - hdict[stations[j]]) ** 2
            D[i, j] = D[j, i] = alpha * DF + (1 - alpha) * DH
    return D


def spectral_cluster(D, n_clusters=3, gamma=None, random_state=0):
    pos = D[D > 0]
    if gamma is None:
        gamma = 1.0 / (np.median(pos) + 1e-12) if pos.size else 1.0

    S = np.exp(-gamma * D)
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    return sc.fit_predict(S)


def best_perm_accuracy(stations, pred_labels):
    true = np.array([TRUE_3CLASS[s] for s in stations])
    C = contingency_matrix(true, pred_labels)
    r, c = linear_sum_assignment(C.max() - C)
    mapping = {c[i]: r[i] for i in range(len(r))}
    mapped = np.array([mapping[l] for l in pred_labels])
    acc = (mapped == true).mean()
    return acc, mapped


FIXED_PARAMS = {
    'year': '2013',
    'feature': 'PM2.5',
    'ell_g': 10.0,
    'ell_i': 4.0,
    'sig2_g': 1.0,
    'sig2_i': 0.5,
    'M_g': 30,
    'M_i': 12,
    'alpha': 0.25,
    'gamma': 8.299952964482483,
    'hours': [2, 14],
    'n_clusters': 3,
    'random_state': 0,
    'noise_sigma2': 0.01,
    'num_rounds': 10,
    'local_iters': 100,
    'lr': 0.1,
    'seed': 0,
}

FS_TITLE = 30
FS_AXIS = 26
FS_TICK = 26
FS_LEGEND = 26

plt.rcParams.update(
    {
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_AXIS,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.weight": "normal",
    }
)


def plot_year_feature_two_hours_3class(
    df,
    year_str,
    feature,
    best,
    stations_order=None,
    title_prefix="pFedHGP 3-way clustering (02:00 and 14:00 only)",
):
    dy = df[df["year_str"] == year_str][["datetime", "station", "date_only", "hour", feature]].dropna()

    if stations_order is None:
        stations_order = sorted(dy["station"].unique(), key=lambda s: s.lower())

    fig, ax = plt.subplots(figsize=(22, 12))

    for s in stations_order:
        d = dy[dy["station"] == s].sort_values("datetime").copy()
        d = d.groupby(["date_only", "hour"], as_index=False).first()
        vals = d[feature].astype(float).values
        t = np.arange(len(vals))

        if len(vals) >= 5:
            ker = np.ones(5) / 5.0
            mean_smooth = np.convolve(np.nan_to_num(vals, nan=np.nanmean(vals)), ker, mode="same")
        else:
            mean_smooth = vals

        win = min(len(vals), 14)
        ser = pd.Series(vals)
        stdv = (
            ser.rolling(win, min_periods=max(4, win // 2))
            .std()
            .bfill()
            .ffill()
            .values
        )

        lab = dict(zip(best["stations"], best["labels"])).get(s, 0)
        c = CMAP3[lab]

        ax.plot(
            t,
            mean_smooth,
            lw=2.8,
            color=c,
            alpha=0.95,
            label=f"{s} ({CLASS_NAMES[lab]})",
        )
        ax.fill_between(t, mean_smooth - stdv, mean_smooth + stdv, color=c, alpha=0.12)

    ax.set_title(f"{title_prefix}: {feature} -- {year_str}", fontsize=FS_TITLE, pad=22)
    ax.set_xlabel("Sample index in the year (02:00 and 14:00)", fontsize=FS_AXIS, labelpad=14)
    ax.set_ylabel(feature, fontsize=FS_AXIS, labelpad=14)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(alpha=0.3)

    ax.legend(
        ncol=3,
        fontsize=FS_LEGEND,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    return fig


def plot_station_map():
    fs_title, fs_axis, fs_tick, fs_label, fs_legend = 29, 27, 23, 20, 23
    with plt.rc_context({
        "font.size": fs_tick,
        "axes.titlesize": fs_title,
        "axes.labelsize": fs_axis,
        "xtick.labelsize": fs_tick,
        "ytick.labelsize": fs_tick,
        "legend.fontsize": fs_legend,
    }):
        fig, ax = plt.subplots(figsize=(13.5, 11.5))

        for st, (lat, lon) in STATION_COORDS.items():
            ax.scatter(
                lon,
                lat,
                s=300,
                c=CMAP3[TRUE_3CLASS[st]],
                edgecolor="black",
                linewidth=1.4,
                zorder=3,
            )
            dx, dy, ha, va = LABEL_OFFSETS[st]
            ax.text(
                lon + dx,
                lat + dy,
                st,
                fontsize=fs_label,
                ha=ha,
                va=va,
                zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            )

        ax.set_title(
            "Geographical Distribution of Beijing Air Quality Stations\n"
            "(True Urban / West-Suburb / East-Suburb)",
            fontsize=fs_title,
            pad=24,
        )
        ax.set_xlabel("Longitude", fontsize=fs_axis, labelpad=12)
        ax.set_ylabel("Latitude", fontsize=fs_axis, labelpad=12)
        ax.tick_params(axis="both", labelsize=fs_tick)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_aspect("equal", adjustable="box")

        lons = [lon for lat, lon in STATION_COORDS.values()]
        lats = [lat for lat, lon in STATION_COORDS.values()]
        ax.set_xlim(min(lons) - 0.10, max(lons) + 0.14)
        ax.set_ylim(min(lats) - 0.09, max(lats) + 0.09)

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=CMAP3[k],
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize=15,
                label=CLASS_NAMES[k],
            )
            for k in [0, 1, 2]
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            fontsize=fs_legend,
            columnspacing=1.5,
            handletextpad=0.6,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def run_experiment(data_pattern=None, output_dir=None):
    root = Path(__file__).resolve().parent
    output_dir = Path(output_dir) if output_dir is not None else root / "climate_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    p = FIXED_PARAMS
    data = load_beijing_local(
        str(data_pattern) if data_pattern is not None else str(root / "beijing" / "PRSA_Data_*.csv"),
        ALL_STATIONS,
    )
    data = data[data["hour"].isin(p["hours"])].copy()
    data["year_str"] = data["datetime"].dt.year.astype(str)
    data["date_only"] = data["datetime"].dt.date
    dfy = data[data["year_str"] == p["year"]].copy()
    Ahat, hdict = build_all_Ahat_h(dfy, p, seed=p["seed"])
    if set(Ahat) != set(ALL_STATIONS):
        raise RuntimeError("The fixed experiment requires valid data for all 12 stations.")
    stations = sorted(Ahat, key=lambda s: s.lower())
    D = pairwise_mats(stations, Ahat, hdict, alpha=p["alpha"])
    labels = spectral_cluster(D, n_clusters=p["n_clusters"], gamma=p["gamma"], random_state=p["random_state"])
    acc, labels = best_perm_accuracy(stations, labels)
    best = dict(p, acc=float(acc), stations=stations, labels=labels)
    fig = plot_year_feature_two_hours_3class(data, p["year"], p["feature"], best, stations_order=stations)
    fig.savefig(output_dir / "clustering_result1.png")
    map_fig = plot_station_map()
    map_fig.savefig(output_dir / "map1.png")
    result = {
        "clusters": {CLASS_NAMES[k]: [s for s, lab in zip(stations, labels) if lab == k] for k in CLASS_NAMES},
        "stations": {s: CLASS_NAMES[lab] for s, lab in zip(stations, labels)},
    }
    (output_dir / "clustering_result1.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.close(fig)
    plt.close(map_fig)

    print(f"\nClustering accuracy: {acc:.3f}")
    for k, name in CLASS_NAMES.items():
        print(f"{name:12s}: {result['clusters'][name]}")
    return best, D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-pattern", help="CSV glob; defaults to beijing/ beside this script")
    parser.add_argument("--output-dir", type=Path, help="Defaults to climate_results/ beside this script")
    args = parser.parse_args()
    run_experiment(args.data_pattern, args.output_dir)


if __name__ == "__main__":
    main()
