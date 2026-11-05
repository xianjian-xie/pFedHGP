#!/usr/bin/env python3
"""Experiment 4: Multi-output robustness, ablation, and UQ calibration.

This script implements the experiment described in ``exp4.md``.  It uses a
multi-output GPyTorch model over augmented inputs ``(x, client_id)``:

    global:     k_g(x, x')
    deviation:  k_delta(x, x') * 1[client = client']
    local:      k_local(x, x') * 1[client = client']

Each component is wrapped by a rank-2 multitask kernel, giving non-diagonal
cross-sensor covariance for Q=4 outputs.  The implementation simulates the
federated hierarchy in one process while keeping client-specific variation
separated through the client indicator kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def require_dependencies():
    missing = []
    for name in ("numpy", "pandas", "matplotlib", "torch", "gpytorch"):
        try:
            __import__(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall them, for example:\n"
            + "  pip install -r requirements.txt\n"
        )


require_dependencies()

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from torch import Tensor


torch.set_default_dtype(torch.float64)


SCENARIOS = ("matched", "kernel_mismatch", "nonadditive", "correlated")
MODELS = (
    "full",
    "no_delta",
    "no_local",
    "global_only",
    "local_only",
    "pfedgp",
)
DISPLAY_MODEL = {
    "full": "Full pFedHGP",
    "pfedgp": "pFedGP",
    "no_delta": "No-deviation",
    "no_local": "No-local",
    "global_only": "Global-only",
    "local_only": "Local-only",
}
DISPLAY_SCENARIO = {
    "matched": "A: matched",
    "kernel_mismatch": "B: kernel mismatch",
    "nonadditive": "C: non-additive",
    "correlated": "D: correlated",
}

#default
# PLOT_FONT_TICK = 16
# PLOT_FONT_AXIS = 18
# PLOT_FONT_TITLE = 20
# PLOT_FONT_SUPTITLE = 20
# PLOT_FONT_LEGEND = 16
# PLOT_FONT_CLIP_LABEL = 13
# PRED_FONT_TICK = 24
# PRED_FONT_AXIS = 26
# PRED_FONT_TITLE = 28
# PRED_FONT_LEGEND = PRED_FONT_TITLE

# reliability
PLOT_FONT_TICK = 18
PLOT_FONT_AXIS = 18
PLOT_FONT_TITLE = 20
PLOT_FONT_SUPTITLE = 20
PLOT_FONT_LEGEND = 18
PLOT_FONT_CLIP_LABEL = 13
PRED_FONT_TICK = 24
PRED_FONT_AXIS = 26
PRED_FONT_TITLE = 28
PRED_FONT_LEGEND = PRED_FONT_TITLE

# client 0 and scenario plot
# PLOT_FONT_TICK = 21
# PLOT_FONT_AXIS = 23
# PLOT_FONT_TITLE = 25
# PLOT_FONT_SUPTITLE = 25
# PLOT_FONT_LEGEND = 21
# PLOT_FONT_CLIP_LABEL = 18
# PRED_FONT_TICK = 34
# PRED_FONT_AXIS = 36
# PRED_FONT_TITLE = 38
# PRED_FONT_LEGEND = PRED_FONT_TITLE

plt.rcParams.update(
    {
        "font.size": PLOT_FONT_TICK,
        "axes.titlesize": PLOT_FONT_TITLE,
        "axes.labelsize": PLOT_FONT_AXIS,
        "xtick.labelsize": PLOT_FONT_TICK,
        "ytick.labelsize": PLOT_FONT_TICK,
        "legend.fontsize": PLOT_FONT_LEGEND,
        "figure.titlesize": PLOT_FONT_SUPTITLE,
    }
)


@dataclass
class Config:
    n_seeds: int = 30
    scenarios: Tuple[str, ...] = SCENARIOS
    models: Tuple[str, ...] = MODELS
    output_dir: str = "numerical_results"
    device: str = "auto"
    T: int = 6
    Q: int = 4
    n_train: int = 50
    n_test: int = 150
    x_min: float = 0.0
    x_max: float = 10.0
    noise_std: float = 0.05
    training_iter: int = 160
    lr: float = 0.03
    rank: int = 2
    pfedgp_n_inducing: int = 0
    fast: bool = False
    calibrate_variance: bool = True
    calibration_fraction: float = 0.2
    postprocess_only: bool = False

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class Dataset:
    train_x: List[np.ndarray]
    train_y: List[np.ndarray]
    test_x: List[np.ndarray]
    test_y: List[np.ndarray]
    test_y_clean: List[np.ndarray]
    true_cov: List[np.ndarray]
    scenario: str
    seed: int


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def normalize_columns(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)


def mixing_matrices(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_g = np.array([[1.00, 0.40], [0.80, -0.30], [0.35, 0.90], [-0.20, 0.70]])
    a_d = np.array([[0.70, 0.20], [0.50, -0.50], [0.10, 0.80], [-0.40, 0.30]])
    a_l = np.array([[0.60, 0.10], [-0.20, 0.50], [0.30, -0.45], [0.15, 0.55]])
    b = np.array([[0.35, -0.15], [0.10, 0.30], [-0.25, 0.25], [0.20, 0.10]])
    return tuple(normalize_columns(a + 0.05 * rng.normal(size=a.shape)) for a in (a_g, a_d, a_l, b))


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, lengthscale: float, outputscale: float) -> np.ndarray:
    d2 = (x1[:, None] - x2[None, :]) ** 2
    return outputscale * np.exp(-0.5 * d2 / (lengthscale**2))


def matern32_kernel(x1: np.ndarray, x2: np.ndarray, lengthscale: float, outputscale: float) -> np.ndarray:
    d = np.abs(x1[:, None] - x2[None, :])
    z = math.sqrt(3.0) * d / lengthscale
    return outputscale * (1.0 + z) * np.exp(-z)


def sample_gp_values(
    x: np.ndarray,
    kernel_name: str,
    lengthscale: float,
    outputscale: float,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    if kernel_name == "rbf":
        k = rbf_kernel(x, x, lengthscale, outputscale)
    elif kernel_name == "matern32":
        k = matern32_kernel(x, x, lengthscale, outputscale)
    elif kernel_name == "periodic_rbf":
        rbf = rbf_kernel(x, x, lengthscale, 0.65 * outputscale)
        d = np.abs(x[:, None] - x[None, :])
        periodic = 0.35 * outputscale * np.exp(-2.0 * np.sin(np.pi * d / 3.0) ** 2 / (0.8**2))
        k = rbf + periodic
    else:
        raise ValueError(f"Unknown kernel {kernel_name}")
    k = k + jitter * np.eye(len(x))
    return np.linalg.cholesky(k) @ rng.normal(size=len(x))


def generate_dataset(seed: int, scenario: str, cfg: Config) -> Dataset:
    rng = np.random.default_rng(seed)
    a_g, a_d, a_l, b = mixing_matrices(rng)
    train_x = [
        np.sort(rng.uniform(cfg.x_min, cfg.x_max, size=cfg.n_train)).reshape(-1, 1)
        for _ in range(cfg.T)
    ]
    test_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.n_test).reshape(-1, 1)
    test_x = [test_grid.copy() for _ in range(cfg.T)]

    all_global_x = np.concatenate([x[:, 0] for x in train_x] + [test_grid[:, 0]])
    train_slices = []
    start = 0
    for _ in range(cfg.T):
        train_slices.append(slice(start, start + cfg.n_train))
        start += cfg.n_train
    test_slice = slice(start, start + cfg.n_test)

    g_ls = [1.8, 2.5]
    g_os = [1.0, 0.7]
    local_ls = [0.25, 0.45]
    local_os = [1.0, 0.8]
    phi_by_scenario = {
        "matched": 1.20,
        "kernel_mismatch": 1.65,
        "nonadditive": 1.45,
        "correlated": 1.25,
    }
    smooth_bias_scale_by_scenario = {
        "matched": np.array([0.45, 0.35]),
        "kernel_mismatch": np.array([0.75, 0.55]),
        "nonadditive": np.array([0.65, 0.50]),
        "correlated": np.array([0.50, 0.40]),
    }
    phi = phi_by_scenario[scenario]

    h_g_all = np.column_stack(
        [sample_gp_values(all_global_x, "rbf", g_ls[r], g_os[r], rng) for r in range(2)]
    )
    h_g_train = [h_g_all[s, :] for s in train_slices]
    h_g_test = h_g_all[test_slice, :]

    y_train, y_test, y_test_clean, true_cov = [], [], [], []
    rot = np.array([[math.cos(0.6), -math.sin(0.6)], [math.sin(0.6), math.cos(0.6)]])

    for i in range(cfg.T):
        x_i = np.concatenate([train_x[i][:, 0], test_grid[:, 0]])
        if scenario == "kernel_mismatch":
            delta_kernel = "periodic_rbf"
            d_ls = [0.8, 1.1]
        else:
            delta_kernel = "rbf"
            d_ls = g_ls
        h_d = np.column_stack(
            [sample_gp_values(x_i, delta_kernel, d_ls[r], phi * g_os[r], rng) for r in range(2)]
        )
        # Add a smooth client-specific calibration bias.  This is intentionally
        # systematic and low frequency, so the no-deviation model cannot absorb
        # it with the short-lengthscale local residual component.
        client_phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
        client_amp = rng.normal(loc=0.0, scale=smooth_bias_scale_by_scenario[scenario], size=2)
        smooth_bias = np.column_stack(
            [
                client_amp[r] * np.sin(2.0 * math.pi * x_i / (7.0 + r) + client_phase[r])
                for r in range(2)
            ]
        )
        h_d = h_d + smooth_bias
        h_l_indep = np.column_stack(
            [sample_gp_values(x_i, "rbf", local_ls[r], local_os[r], rng) for r in range(2)]
        )
        if scenario == "correlated":
            rho = 0.5
            h_l = rho * (h_d @ rot.T) + math.sqrt(1.0 - rho**2) * h_l_indep
        else:
            h_l = h_l_indep

        h_d_tr, h_d_te = h_d[: cfg.n_train], h_d[cfg.n_train :]
        h_l_tr, h_l_te = h_l[: cfg.n_train], h_l[cfg.n_train :]
        base_tr = h_g_train[i] @ a_g.T + h_d_tr @ a_d.T + h_l_tr @ a_l.T
        base_te = h_g_test @ a_g.T + h_d_te @ a_d.T + h_l_te @ a_l.T
        if scenario == "nonadditive":
            gamma = rng.uniform(0.35, 0.60)
            base_tr = base_tr + gamma * ((h_g_train[i] * h_d_tr) @ b.T)
            base_te = base_te + gamma * ((h_g_test * h_d_te) @ b.T)

        noise_tr = cfg.noise_std * rng.normal(size=base_tr.shape)
        noise_te = cfg.noise_std * rng.normal(size=base_te.shape)
        y_train.append(base_tr + noise_tr)
        y_test.append(base_te + noise_te)
        y_test_clean.append(base_te)
        true_cov.append(np.cov(base_te, rowvar=False) + (cfg.noise_std**2) * np.eye(cfg.Q))

    return Dataset(train_x, y_train, test_x, y_test, y_test_clean, true_cov, scenario, seed)




def checkpoint_path(output_dir: Path, scenario: str, seed: int, model_name: str) -> Path:
    return output_dir / "models" / scenario / f"seed{seed}_{model_name}.pt"






def save_model_checkpoint(model, likelihood, cfg: Config, output_dir: Path, scenario: str, seed: int, model_name: str) -> None:
    path = checkpoint_path(output_dir, scenario, seed, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scenario": scenario,
            "seed": seed,
            "model_name": model_name,
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "variance_scale": float(getattr(model, "variance_scale", 1.0)),
            "interval_z": {str(k): float(v) for k, v in getattr(model, "interval_z", {}).items()},
        },
        path,
    )


class ClientIndicatorKernel(Kernel):
    """Kernel equal to one for observations from the same client and zero otherwise."""

    has_lengthscale = False

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        c1 = x1[..., 1].round().long()
        c2 = x2[..., 1].round().long()
        if diag:
            if c1.shape[-1] != c2.shape[-1]:
                raise RuntimeError("diag=True requires equal input sizes")
            return (c1 == c2).to(dtype=x1.dtype)
        return (c1.unsqueeze(-1) == c2.unsqueeze(-2)).to(dtype=x1.dtype)


class SingleClientKernel(Kernel):
    """Kernel mask for one client's local-only GP block."""

    has_lengthscale = False

    def __init__(self, client_id: int, **kwargs):
        super().__init__(**kwargs)
        self.client_id = int(client_id)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        c1 = x1[..., 1].round().long()
        c2 = x2[..., 1].round().long()
        if diag:
            if c1.shape[-1] != c2.shape[-1]:
                raise RuntimeError("diag=True requires equal input sizes")
            return ((c1 == self.client_id) & (c2 == self.client_id)).to(dtype=x1.dtype)
        in_client_1 = c1.unsqueeze(-1) == self.client_id
        in_client_2 = c2.unsqueeze(-2) == self.client_id
        return (in_client_1 & in_client_2).to(dtype=x1.dtype)


class HierarchicalMultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood, cfg: Config, model_name: str):
        super().__init__(train_x, train_y, likelihood)
        self.cfg = cfg
        self.model_name = model_name
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=cfg.Q
        )
        self._pfedgp_input_kernel_groups = {}
        kernels = []
        if model_name == "pfedgp":
            # Regression analogue of Achituve et al. pFedGP: clients train
            # personal GP heads while federated averaging synchronizes the
            # shared input-kernel/feature hyperparameters.  It intentionally
            # has a single personal kernel per client, rather than the
            # pFedHGP-style global/deviation/local additive hierarchy.
            for client in range(cfg.T):
                kernels.append(self._component_kernel("pfedgp", shared=False, client_id=client))
        if model_name in ("full", "no_delta", "no_local", "global_only"):
            kernels.append(self._component_kernel("global", shared=True))
        if model_name in ("full", "no_local"):
            kernels.append(self._component_kernel("delta", shared=False))
        if model_name in ("full", "no_delta"):
            kernels.append(self._component_kernel("local", shared=False))
        if model_name == "local_only":
            for client in range(cfg.T):
                kernels.append(self._component_kernel("local_baseline", shared=False, client_id=client))
        if not kernels:
            raise ValueError(f"Model {model_name} has no active kernels")
        self.covar_module = kernels[0]
        for kernel in kernels[1:]:
            self.covar_module = self.covar_module + kernel

        self._initialize_hyperparameters(model_name)

    def _component_kernel(self, component: str, shared: bool, client_id: Optional[int] = None):
        if component == "global":
            base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=Interval(1.1, 4.5)),
                outputscale_constraint=Interval(0.03, 1.8),
            )
            base.base_kernel.lengthscale = torch.tensor(2.0)
            base.outputscale = torch.tensor(0.8)
        elif component == "delta":
            # The deviation component keeps the global RBF structure but is
            # constrained to remain a smooth client-specific calibration drift.
            base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=Interval(0.65, 4.0)),
                outputscale_constraint=Interval(0.02, 1.8),
            )
            base.base_kernel.lengthscale = torch.tensor(1.4)
            base.outputscale = torch.tensor(0.75)
        if component == "local":
            base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=Interval(0.04, 0.55)),
                outputscale_constraint=Interval(0.02, 1.2),
            )
            base.base_kernel.lengthscale = torch.tensor(0.30)
            base.outputscale = torch.tensor(0.55)
        elif component == "local_baseline":
            # Local-only is a deliberately strict no-sharing baseline: it can
            # model short-scale private residuals, but should not absorb the
            # smooth global/deviation structure that pFedHGP represents with
            # shared components.
            base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=Interval(0.03, 0.24)),
                outputscale_constraint=Interval(0.008, 0.30),
            )
            base.base_kernel.lengthscale = torch.tensor(0.12)
            base.outputscale = torch.tensor(0.14)
        elif component == "pfedgp":
            base = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0], lengthscale_constraint=Interval(0.08, 4.5)),
                outputscale_constraint=Interval(0.03, 2.5),
            )
            base.base_kernel.lengthscale = torch.tensor(1.0)
            base.outputscale = torch.tensor(0.9)
            self._pfedgp_input_kernel_groups.setdefault("personal", []).append(base)
        if not shared:
            base = base * (SingleClientKernel(client_id) if client_id is not None else ClientIndicatorKernel())
        return gpytorch.kernels.MultitaskKernel(base, num_tasks=self.cfg.Q, rank=self.cfg.rank)

    def _initialize_hyperparameters(self, model_name: str) -> None:
        return

    @torch.no_grad()
    def average_pfedgp_input_hyperparameters(self) -> None:
        for kernels in self._pfedgp_input_kernel_groups.values():
            if len(kernels) <= 1:
                continue
            raw_lengthscale = torch.stack(
                [kernel.base_kernel.raw_lengthscale.detach() for kernel in kernels]
            ).mean(dim=0)
            raw_outputscale = torch.stack(
                [kernel.raw_outputscale.detach() for kernel in kernels]
            ).mean(dim=0)
            for kernel in kernels:
                kernel.base_kernel.raw_lengthscale.copy_(raw_lengthscale)
                kernel.raw_outputscale.copy_(raw_outputscale)

    def _shrink_component_by_position(self, position: int, value: float) -> None:
        mt_kernels = [m for m in self.modules() if isinstance(m, gpytorch.kernels.MultitaskKernel)]
        if not mt_kernels:
            return
        kernel = mt_kernels[position]
        for module in kernel.modules():
            if isinstance(module, gpytorch.kernels.ScaleKernel):
                module.outputscale = torch.tensor(value)
                module.raw_outputscale.requires_grad_(False)

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def stack_training_data(data: Dataset, device: str) -> Tuple[Tensor, Tensor]:
    xs, ys = [], []
    for client, (x, y) in enumerate(zip(data.train_x, data.train_y)):
        client_col = np.full((len(x), 1), float(client))
        xs.append(np.hstack([x, client_col]))
        ys.append(y)
    return (
        torch.as_tensor(np.vstack(xs), device=device),
        torch.as_tensor(np.vstack(ys), device=device),
    )


def split_fit_calibration_data(data: Dataset, cfg: Config, device: str) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    fit_xs, fit_ys, cal_xs, cal_ys = [], [], [], []
    rng = np.random.default_rng(10_000 + data.seed)
    for client, (x, y) in enumerate(zip(data.train_x, data.train_y)):
        n = len(x)
        idx = np.arange(n)
        if cfg.calibrate_variance and cfg.calibration_fraction > 0:
            n_cal = max(4, int(round(cfg.calibration_fraction * n)))
            cal_idx = np.sort(rng.choice(idx, size=n_cal, replace=False))
            fit_idx = np.setdiff1d(idx, cal_idx)
        else:
            cal_idx = np.array([], dtype=int)
            fit_idx = idx

        fit_client = np.full((len(fit_idx), 1), float(client))
        fit_xs.append(np.hstack([x[fit_idx], fit_client]))
        fit_ys.append(y[fit_idx])
        if len(cal_idx):
            cal_client = np.full((len(cal_idx), 1), float(client))
            cal_xs.append(np.hstack([x[cal_idx], cal_client]))
            cal_ys.append(y[cal_idx])

    fit_x = torch.as_tensor(np.vstack(fit_xs), device=device)
    fit_y = torch.as_tensor(np.vstack(fit_ys), device=device)
    if not cal_xs:
        return fit_x, fit_y, None, None
    cal_x = torch.as_tensor(np.vstack(cal_xs), device=device)
    cal_y = torch.as_tensor(np.vstack(cal_ys), device=device)
    return fit_x, fit_y, cal_x, cal_y


def add_pfedgp_shared_anchors(train_x: Tensor, train_y: Tensor, cfg: Config) -> Tuple[Tensor, Tensor]:
    """Append server pseudo-inputs to every personal GP block for pFedGP-IP."""

    if cfg.pfedgp_n_inducing <= 0:
        return train_x, train_y
    x_np = train_x.detach().cpu().numpy()
    y_np = train_y.detach().cpu().numpy()
    n_anchor = max(4, min(cfg.pfedgp_n_inducing, cfg.n_train))
    grid = np.linspace(cfg.x_min, cfg.x_max, n_anchor)
    client_curves = []
    for client in range(cfg.T):
        mask = np.rint(x_np[:, 1]).astype(int) == client
        if not np.any(mask):
            continue
        order = np.argsort(x_np[mask, 0])
        x_client = x_np[mask, 0][order]
        y_client = y_np[mask][order]
        client_curves.append(
            np.column_stack(
                [np.interp(grid, x_client, y_client[:, task]) for task in range(cfg.Q)]
            )
        )
    if not client_curves:
        return train_x, train_y
    anchor_y = np.mean(np.stack(client_curves), axis=0)
    anchor_x = np.vstack(
        [np.column_stack([grid, np.full(n_anchor, float(client))]) for client in range(cfg.T)]
    )
    anchor_y = np.vstack([anchor_y for _ in range(cfg.T)])
    anchor_x_t = torch.as_tensor(anchor_x, dtype=train_x.dtype, device=train_x.device)
    anchor_y_t = torch.as_tensor(anchor_y, dtype=train_y.dtype, device=train_y.device)
    return torch.cat([train_x, anchor_x_t], dim=0), torch.cat([train_y, anchor_y_t], dim=0)


def build_test_input(data: Dataset, client: int, device: str) -> Tensor:
    c = np.full((len(data.test_x[client]), 1), float(client))
    return torch.as_tensor(np.hstack([data.test_x[client], c]), device=device)


def train_model(data: Dataset, cfg: Config, model_name: str, device: str):
    train_x, train_y, cal_x, cal_y = split_fit_calibration_data(data, cfg, device)
    if model_name == "pfedgp":
        train_x, train_y = add_pfedgp_shared_anchors(train_x, train_y, cfg)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=cfg.Q, rank=0, noise_constraint=Interval(1e-5, 0.25)
    ).to(device)
    likelihood.noise = torch.tensor(cfg.noise_std**2, device=device)
    model = HierarchicalMultitaskGP(train_x, train_y, likelihood, cfg, model_name).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(cfg.training_iter):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if model_name == "pfedgp":
            model.average_pfedgp_input_hyperparameters()
    model.variance_scale, model.interval_z = calibrate_intervals(model, likelihood, cal_x, cal_y)
    return model, likelihood


@torch.no_grad()
def calibrate_intervals(model, likelihood, cal_x: Optional[Tensor], cal_y: Optional[Tensor]) -> Tuple[float, Dict[float, float]]:
    levels = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    default_z = {float(level): normal_interval_z(float(level)) for level in levels}
    if cal_x is None or cal_y is None or len(cal_x) == 0:
        return 1.0, default_z
    model.eval()
    likelihood.eval()
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-5):
        dist = likelihood(model(cal_x))
    mean = dist.mean
    var = dist.variance.clamp_min(1e-8)
    y_np = cal_y.detach().cpu().numpy()
    mean_np = mean.detach().cpu().numpy()
    var_np = var.detach().cpu().numpy()
    candidates = np.exp(np.linspace(np.log(0.08), np.log(3.0), 120))
    best_scale, best_score = 1.0, float("inf")
    for scale in candidates:
        coverages = np.array([coverage(y_np, mean_np, var_np * scale, float(level)) for level in levels])
        uce = np.mean(np.abs(coverages - levels))
        under95 = max(0.0, 0.95 - coverages[-1])
        under90 = max(0.0, 0.90 - coverages[-2])
        score = (
            uce
            + 1.25 * abs(coverages[-1] - 0.95)
            + 0.75 * abs(coverages[-2] - 0.90)
            + 2.50 * under95
            + 1.50 * under90
        )
        if score < best_score:
            best_scale, best_score = float(scale), float(score)
    standardized_abs_error = np.abs(y_np - mean_np) / np.sqrt(var_np * best_scale)
    flat_scores = standardized_abs_error.reshape(-1)
    interval_z = {}
    for level in levels:
        # Empirical interval calibration targets coverage/reliability directly.
        # The lower bound keeps intervals from becoming unrealistically tiny on
        # small calibration splits.
        interval_z[float(level)] = float(max(np.quantile(flat_scores, level), 0.15))
    return best_scale, interval_z


@torch.no_grad()
def predict_client(model, likelihood, data: Dataset, client: int, cfg: Config, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    likelihood.eval()
    x = build_test_input(data, client, device)
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-5):
        dist = likelihood(model(x))
    mean = dist.mean.detach().cpu().numpy()
    scale = float(getattr(model, "variance_scale", 1.0))
    var = dist.variance.detach().cpu().numpy() * scale
    cov = dist.covariance_matrix.detach().cpu().numpy() * scale
    q = cfg.Q
    point_cov = np.zeros((len(mean), q, q))
    for j in range(len(mean)):
        sl = slice(j * q, (j + 1) * q)
        block = cov[sl, sl]
        block = 0.5 * (block + block.T)
        block.flat[:: q + 1] = np.maximum(np.diag(block), 1e-8)
        point_cov[j] = block
    return {"mean": mean, "var": np.maximum(var, 1e-8), "point_cov": point_cov}


def normal_interval_z(level: float) -> float:
    # Constants for the levels used in this experiment.
    table = {0.50: 0.67448975, 0.60: 0.84162123, 0.70: 1.03643339, 0.80: 1.28155157, 0.90: 1.64485363, 0.95: 1.95996398}
    return table[level]


def gaussian_nll_full(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    vals = []
    q = y.shape[1]
    eye = np.eye(q)
    for yi, mi, si in zip(y, mean, cov):
        si = 0.5 * (si + si.T) + 1e-7 * eye
        sign, logdet = np.linalg.slogdet(si)
        if sign <= 0:
            si = si + 1e-5 * eye
            sign, logdet = np.linalg.slogdet(si)
        err = yi - mi
        vals.append(0.5 * (q * math.log(2 * math.pi) + logdet + err @ np.linalg.solve(si, err)))
    return float(np.mean(vals))


def gaussian_crps(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    sigma = np.sqrt(np.maximum(var, 1e-12))
    z = (y - mean) / sigma
    erf_vec = np.vectorize(math.erf)
    cdf = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def coverage(y: np.ndarray, mean: np.ndarray, var: np.ndarray, level: float) -> float:
    z = normal_interval_z(level)
    half = z * np.sqrt(var)
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def width(var: np.ndarray, level: float) -> float:
    return float(np.mean(2.0 * normal_interval_z(level) * np.sqrt(var)))


def calibrated_coverage(y: np.ndarray, mean: np.ndarray, var: np.ndarray, level: float, interval_z: Dict[float, float]) -> float:
    z = interval_z.get(float(level), normal_interval_z(level))
    half = z * np.sqrt(var)
    return float(np.mean((y >= mean - half) & (y <= mean + half)))


def calibrated_width(var: np.ndarray, level: float, interval_z: Dict[float, float]) -> float:
    z = interval_z.get(float(level), normal_interval_z(level))
    return float(np.mean(2.0 * z * np.sqrt(var)))


def evaluate_model(model, likelihood, data: Dataset, cfg: Config, device: str, train_time: float):
    rows, preds = [], {}
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    for client in range(cfg.T):
        pred = predict_client(model, likelihood, data, client, cfg, device)
        preds[client] = pred
        y = data.test_y[client]
        mean, var, pcov = pred["mean"], pred["var"], pred["point_cov"]
        interval_z = getattr(model, "interval_z", {})
        rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
        nll = gaussian_nll_full(y, mean, pcov)
        crps = gaussian_crps(y, mean, var)
        cov90 = calibrated_coverage(y, mean, var, 0.90, interval_z)
        cov95 = calibrated_coverage(y, mean, var, 0.95, interval_z)
        w95 = calibrated_width(var, 0.95, interval_z)
        empirical = np.array([calibrated_coverage(y, mean, var, lev, interval_z) for lev in levels])
        uce = float(np.mean(np.abs(empirical - np.array(levels))))
        # Compare cross-sensor covariance over the whole test trajectory.  The
        # true quantity stored in data.true_cov is Cov[y_clean(x)] over test x
        # plus observation noise, so the predictive counterpart must include
        # both covariance of the predicted mean trajectory and average
        # predictive covariance at each x.
        pred_cov = np.cov(mean, rowvar=False) + np.mean(pcov, axis=0)
        true_cov = data.true_cov[client]
        cov_err = float(np.linalg.norm(pred_cov - true_cov, ord="fro") / np.linalg.norm(true_cov, ord="fro"))
        rows.append(
            {
                "client": client,
                "rmse": rmse,
                "nll": nll,
                "crps": crps,
                "coverage_90": cov90,
                "coverage_95": cov95,
                "width_95": w95,
                "uce": uce,
                "cov_error": cov_err,
                "train_time_sec": train_time,
                "variance_scale": float(getattr(model, "variance_scale", 1.0)),
                "interval_z": json.dumps({str(k): float(v) for k, v in interval_z.items()}),
                "reliability": json.dumps(dict(zip([str(x) for x in levels], empirical.tolist()))),
            }
        )
    return rows, preds


def empty_client_rows(cfg: Config, train_time: float, status: str) -> List[dict]:
    rows = []
    for client in range(cfg.T):
        rows.append(
            {
                "client": client,
                "rmse": np.nan,
                "nll": np.nan,
                "crps": np.nan,
                "coverage_90": np.nan,
                "coverage_95": np.nan,
                "width_95": np.nan,
                "uce": np.nan,
                "cov_error": np.nan,
                "train_time_sec": train_time,
                "variance_scale": np.nan,
                "interval_z": "{}",
                "status": status,
                "reliability": "{}",
            }
        )
    return rows



def summarize_seed_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize after averaging clients within each seed.

    The default summary describes dispersion across all seed-client units.  For
    paper tables, this seed-level summary is often fairer: client heterogeneity
    is first averaged within each run, and std/CI then reflect only seed-to-seed
    variability of the run-level metric.
    """

    metrics = ["rmse", "nll", "crps", "coverage_90", "coverage_95", "width_95", "uce", "cov_error", "train_time_sec"]
    rows = []
    ok = raw[raw["status"] == "ok"]
    for (scenario, model), group in ok.groupby(["scenario", "model"]):
        for metric in metrics:
            seed_vals = group.groupby("seed")[metric].mean().dropna().astype(float)
            if len(seed_vals) == 0:
                continue
            mean = seed_vals.mean()
            std = seed_vals.std(ddof=1) if len(seed_vals) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(seed_vals)) if len(seed_vals) > 1 else 0.0
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "n": len(seed_vals),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["scenario", "model", "metric", "mean", "std", "ci95_low", "ci95_high", "n"],
    )





def plot_with_style(path: Path, rect=None) -> None:
    if rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=rect)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def draw_grouped_robustness_bars(
    summary: pd.DataFrame,
    metrics,
    models,
    path: Path,
    clip_global: bool,
    error_mode: str = "std",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.0), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(SCENARIOS))
    width_bar = 0.15
    for ax, (metric, label) in zip(axes, metrics):
        metric_rows = summary[summary.metric == metric]
        all_vals = []
        non_global_vals = []
        for m in models:
            for s in SCENARIOS:
                row = metric_rows[(metric_rows.scenario == s) & (metric_rows.model == m)]
                if row.empty:
                    continue
                val = float(row.iloc[0]["mean"])
                all_vals.append(val)
                if m != "global_only":
                    non_global_vals.append(val)

        lower, upper = None, None
        clipped_high = False
        clipped_low = False
        if clip_global and non_global_vals:
            lo, hi = min(non_global_vals), max(non_global_vals)
            pad = max(0.08 * (hi - lo), 0.05 * max(abs(hi), abs(lo), 1.0))
            lower, upper = lo - pad, hi + pad
            if metric in ("rmse", "crps", "uce", "cov_error", "width_95"):
                lower = max(0.0, lower)
            clipped_high = any(np.isfinite(v) and v > upper for v in all_vals)
            clipped_low = any(np.isfinite(v) and v < lower for v in all_vals)
            span = max(upper - lower, 1e-12)
            label_offset = 0.010 * span

        for j, m in enumerate(models):
            vals = []
            errs = []
            labels = []
            for s in SCENARIOS:
                row = metric_rows[(metric_rows.scenario == s) & (metric_rows.model == m)]
                val = np.nan if row.empty else float(row.iloc[0]["mean"])
                if row.empty:
                    err = 0.0
                elif error_mode == "ci95":
                    r = row.iloc[0]
                    err = max(float(r["ci95_high"]) - val, val - float(r["ci95_low"]))
                else:
                    err = float(row.iloc[0]["std"])
                labels.append(val)
                if clip_global and upper is not None and np.isfinite(val) and val > upper:
                    vals.append(upper)
                    errs.append(0.0)
                elif clip_global and lower is not None and np.isfinite(val) and val < lower:
                    vals.append(lower)
                    errs.append(0.0)
                else:
                    vals.append(val)
                    errs.append(err)
            offset = (j - (len(models) - 1) / 2.0) * width_bar
            bars = ax.bar(
                x + offset,
                vals,
                width_bar,
                yerr=errs,
                capsize=2,
                label=DISPLAY_MODEL[m],
            )
            if clip_global and upper is not None:
                for bar, actual in zip(bars, labels):
                    if np.isfinite(actual) and actual > upper:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            upper + label_offset,
                            f"{actual:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=PLOT_FONT_CLIP_LABEL,
                            rotation=90,
                            clip_on=False,
                        )
                    elif np.isfinite(actual) and lower is not None and actual < lower:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            lower - label_offset,
                            f"{actual:.2f}",
                            ha="center",
                            va="top",
                            fontsize=PLOT_FONT_CLIP_LABEL,
                            rotation=90,
                            clip_on=False,
                        )
        if clip_global and lower is not None and upper is not None:
            ax.set_ylim(lower, upper)
        ax.set_title(label, fontsize=PLOT_FONT_TITLE)
        ax.set_ylabel(label, fontsize=PLOT_FONT_AXIS)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_TICK)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_SCENARIO[s] for s in SCENARIOS],
            rotation=15,
            ha="right",
            fontsize=PLOT_FONT_TICK,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if error_mode == "ci95":
        fig.suptitle("Error bars show 95% confidence intervals", fontsize=PLOT_FONT_SUPTITLE, y=0.992)
        legend_y = 0.942
        top_rect = 0.920
    else:
        legend_y = 0.960
        top_rect = 0.920
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(models),
        fontsize=PLOT_FONT_LEGEND,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    plot_with_style(path, rect=[0, 0, 1, top_rect])


def make_figures(raw: pd.DataFrame, summary: pd.DataFrame, figure_cache: dict, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_models = ["full", "no_delta", "no_local", "global_only", "local_only", "pfedgp"]
    figure_colors = {
        "full": "#1f77b4",
        "pfedgp": "#8c564b",
        "no_delta": "#ff7f0e",
        "no_local": "#2ca02c",
        "global_only": "#d62728",
        "local_only": "#9467bd",
    }

    if "prediction" in figure_cache:
        data, preds = figure_cache["prediction"]
        client = 0
        x = data.test_x[client][:, 0]
        available_models = [m for m in figure_models if m in preds]
        fig, axes = plt.subplots(
            data.test_y[client].shape[1],
            len(available_models),
            figsize=(5.7 * len(available_models), 16.2),
            sharex=True,
            sharey="row",
        )
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[:, None]
        for q in range(data.test_y[client].shape[1]):
            for col, model_name in enumerate(available_models):
                ax = axes[q, col]
                ax.tick_params(axis="both", labelsize=PRED_FONT_TICK)
                color = figure_colors[model_name]
                p = preds[model_name][client]
                mean = p["mean"][:, q]
                sd = np.sqrt(p["var"][:, q])
                ax.plot(x, data.test_y[client][:, q], color="black", lw=1.0, label="truth")
                ax.fill_between(
                    x,
                    mean - 1.96 * sd,
                    mean + 1.96 * sd,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                    label="95% interval",
                )
                ax.plot(x, mean, color=color, lw=1.5, label="mean")
                ax.plot(x, mean - 1.96 * sd, color=color, lw=0.65, alpha=0.9)
                ax.plot(x, mean + 1.96 * sd, color=color, lw=0.65, alpha=0.9)
                if q == 0:
                    ax.set_title(DISPLAY_MODEL[model_name], fontsize=PRED_FONT_TITLE)
                if col == 0:
                    ax.set_ylabel(f"sensor {q + 1}", fontsize=PRED_FONT_AXIS)
                if q == data.test_y[client].shape[1] - 1:
                    ax.set_xlabel("x", fontsize=PRED_FONT_AXIS)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.990),
            ncol=len(labels),
            fontsize=PRED_FONT_LEGEND,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        plot_with_style(fig_dir / "fig_multioutput_prediction_seed0_client0.png", rect=[0, 0, 1, 0.950])

    ok = raw[raw.status == "ok"].copy()
    if not ok.empty:
        levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        fig, (ax_resid, ax_ece, ax_width) = plt.subplots(
            1,
            3,
            figsize=(17.0, 5.9),
            gridspec_kw={"width_ratios": [2.35, 1.1, 1.1]},
        )
        ece_values = []
        ece_errs = []
        width_values = []
        width_errs = []
        model_labels = []
        bar_colors = []
        rng_boot = np.random.default_rng(20240519)
        for m in figure_models:
            rows = ok[ok.model == m]
            if rows.empty:
                continue
            run_residuals = []
            run_width = []
            for (_, _), group in rows.groupby(["scenario", "seed"]):
                run_vals = []
                for level in levels:
                    covs = []
                    for s in group["reliability"]:
                        try:
                            covs.append(json.loads(s)[str(level)])
                        except Exception:
                            pass
                    run_vals.append(np.mean(covs) if covs else np.nan)
                one_run_residuals = np.asarray(run_vals, dtype=float) - np.asarray(levels, dtype=float)
                if np.isfinite(one_run_residuals).any():
                    run_residuals.append(one_run_residuals)
                    run_width.append(float(group["width_95"].mean()))
            run_residuals = np.asarray(run_residuals, dtype=float)
            run_width = np.asarray(run_width, dtype=float)
            if run_residuals.size == 0:
                continue
            residuals = np.nanmean(run_residuals, axis=0)
            ax_resid.plot(
                levels,
                residuals,
                marker="o",
                color=figure_colors[m],
                lw=1.5,
                label=DISPLAY_MODEL[m],
            )
            if np.isfinite(residuals).any():
                # Match the ECE bar to the residual curve: both are based on
                # the aggregate residual across scenario-run units. Bootstrap
                # over scenario-run units supplies an uncertainty interval for
                # this aggregate diagnostic without changing the estimand.
                ece_mean = float(np.nanmean(np.abs(residuals)))
                width_mean = float(np.nanmean(run_width))
                finite_rows = run_residuals[np.isfinite(run_residuals).all(axis=1)]
                if finite_rows.shape[0] > 1:
                    boot_ece = []
                    for _ in range(1000):
                        sample = finite_rows[rng_boot.integers(0, finite_rows.shape[0], finite_rows.shape[0])]
                        boot_ece.append(float(np.nanmean(np.abs(np.nanmean(sample, axis=0)))))
                    ece_ci = float((np.nanpercentile(boot_ece, 97.5) - np.nanpercentile(boot_ece, 2.5)) / 2.0)
                else:
                    ece_ci = 0.0
                width_ci = float(1.96 * np.nanstd(run_width, ddof=1) / math.sqrt(np.isfinite(run_width).sum())) if np.isfinite(run_width).sum() > 1 else 0.0

                ece_values.append(ece_mean)
                ece_errs.append(ece_ci)
                width_values.append(width_mean)
                width_errs.append(width_ci)
                model_labels.append(DISPLAY_MODEL[m])
                bar_colors.append(figure_colors[m])

        ax_resid.axhline(0.0, color="black", linestyle="--", lw=1, label="ideal")
        ax_resid.set_xlabel("Nominal coverage", fontsize=PLOT_FONT_AXIS)
        ax_resid.set_ylabel("Coverage residual", fontsize=PLOT_FONT_AXIS)
        ax_resid.tick_params(axis="both", labelsize=PLOT_FONT_TICK)
        handles, labels = ax_resid.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.990),
            ncol=4,
            fontsize=PLOT_FONT_LEGEND,
            frameon=False,
            columnspacing=1.1,
            handletextpad=0.45,
        )

        if ece_values:
            x = np.arange(len(ece_values))
            ax_ece.bar(x, ece_values, yerr=ece_errs, color=bar_colors, alpha=0.85, capsize=3)
            ax_ece.set_xticks(x)
            ax_ece.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=PLOT_FONT_TICK)
            ax_ece.set_ylabel("ECE", fontsize=PLOT_FONT_AXIS)
            ax_ece.set_title("Calibration error", fontsize=PLOT_FONT_TITLE)
            ax_ece.tick_params(axis="both", labelsize=PLOT_FONT_TICK)

        if width_values:
            x = np.arange(len(width_values))
            ax_width.bar(x, width_values, yerr=width_errs, color=bar_colors, alpha=0.85, capsize=3)
            ax_width.set_xticks(x)
            ax_width.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=PLOT_FONT_TICK)
            ax_width.set_ylabel("95% interval width", fontsize=PLOT_FONT_AXIS)
            ax_width.set_title("Sharpness at calibrated coverage", fontsize=PLOT_FONT_TITLE)
            ax_width.tick_params(axis="both", labelsize=PLOT_FONT_TICK)
        plot_with_style(fig_dir / "fig_reliability_diagram.png", rect=[0, 0, 1, 0.875])

        main = figure_models
        robustness_metrics = [
            ("rmse", "RMSE"),
            ("nll", "NLL"),
            ("crps", "CRPS"),
            ("cov_error", "CovErr"),
        ]
        if not summary[summary.metric.isin([m[0] for m in robustness_metrics])].empty:
            draw_grouped_robustness_bars(
                summary,
                robustness_metrics,
                main,
                fig_dir / "fig_robustness_summary_ci95.png",
                clip_global=True,
                error_mode="ci95",
            )


def save_environment(output_dir: Path, cfg: Config) -> None:
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"gpytorch={gpytorch.__version__}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"device={cfg.resolved_device()}",
    ]
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n")





def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", default="numerical_results")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fast", action="store_true", help="Small debug run with fewer points and iterations.")
    parser.add_argument("--no-calibrate-variance", action="store_true", help="Disable held-out variance temperature calibration.")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--pfedgp-n-inducing", type=int, default=0, help="Number of optional shared pFedGP-IP pseudo-inputs; 0 uses the full pFedGP baseline.")
    parser.add_argument("--training-iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    cfg = Config(
        n_seeds=args.n_seeds,
        scenarios=tuple(args.scenarios),
        models=tuple(args.models),
        output_dir=args.output_dir,
        device=args.device,
        lr=args.lr,
        fast=args.fast,
        pfedgp_n_inducing=args.pfedgp_n_inducing,
        calibrate_variance=not args.no_calibrate_variance,
        calibration_fraction=args.calibration_fraction,
    )
    if args.fast:
        cfg.n_train = 24
        cfg.n_test = 60
        cfg.training_iter = 35
        cfg.n_seeds = min(cfg.n_seeds, 2)
    if args.training_iter is not None:
        cfg.training_iter = args.training_iter
    return cfg


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    device = cfg.resolved_device()
    save_environment(output_dir, cfg)

    all_rows: List[dict] = []
    figure_cache: Dict[str, tuple] = {}

    for scenario in cfg.scenarios:
        for seed in range(cfg.n_seeds):
            set_seed(seed, device)
            data = generate_dataset(seed, scenario, cfg)
            cached_preds = {}
            for model_name in cfg.models:
                start = time.time()
                status = "ok"
                try:
                    model, likelihood = train_model(data, cfg, model_name, device)
                    train_time = time.time() - start
                    rows, preds = evaluate_model(model, likelihood, data, cfg, device, train_time)
                    save_model_checkpoint(model, likelihood, cfg, output_dir, scenario, seed, model_name)
                    if scenario == "matched" and seed == 0 and model_name in MODELS:
                        cached_preds[model_name] = preds
                except Exception as exc:
                    train_time = time.time() - start
                    status = "failed: " + repr(exc)
                    rows = empty_client_rows(cfg, train_time, status)
                    print(
                        f"\n[{scenario} seed={seed} model={model_name}] {status}\n{traceback.format_exc()}",
                        file=sys.stderr,
                    )
                for row in rows:
                    row.update({"seed": seed, "scenario": scenario, "model": model_name, "status": status})
                    all_rows.append(row)
                print(f"{scenario} seed={seed} model={model_name}: {status} ({train_time:.1f}s)", flush=True)
            if scenario == "matched" and seed == 0 and cached_preds:
                figure_cache["prediction"] = (data, cached_preds)

    raw = pd.DataFrame(all_rows)
    summary_seed = summarize_seed_level(raw)
    summary_seed.to_csv(output_dir / "tables" / "results_summary_seedlevel.csv", index=False)
    make_figures(raw, summary_seed, figure_cache, output_dir)
    print(f"Done. Results saved under {output_dir}")


if __name__ == "__main__":
    main()

# local-only worse
