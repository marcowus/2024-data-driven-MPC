import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class MPCConfig:
    # Horizons
    h_p: int = 15
    h_c: int = 7

    # Weights
    q_state_diag: tuple = (100.0, 1000.0, 100.0)
    r_control_diag: tuple = (1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4)
    lambda_reg: float = 1e-3

    # Identification & robustness
    forgetting_factor: float = 0.995
    disturbance_quantile: float = 0.95
    disturbance_scale: float = 1.2
    robust_tightening_scale: float = 1.0

    # Simulation
    sim_steps: int = 120
    random_seed: int = 2026

    # Input envelopes
    max_deviation_from_historical_percent: float = 0.10
    min_abs_deviation_allowance: tuple = (5, 50, 0.5, 1, 1, 0.1, 10)


STATES_NAMES_CN = ["松散回潮-热风温度", "松散回潮-出料含水率", "松散回潮-出料温度"]
INPUTS_NAMES_CN = [
    "松散回潮-工艺流量实际值",
    "松散回潮-罩压力",
    "松散回潮-加水流量",
    "松散回潮-汽水混合自动阀门开度",
    "松散回潮-蒸汽自动阀门开度",
    "松散回潮-入口含水率",
    "松散回潮-加水累计量",
]

STATES_NAMES_EN = ["Furnace Temperature", "Outlet Moisture", "Outlet Temperature"]
INPUTS_NAMES_EN = [
    "Process Flow Rate",
    "Hood Pressure",
    "Water Flow",
    "Water-Steam Mix Valve",
    "Steam Valve",
    "Inlet Moisture",
    "Cumulative Water",
]


class WeightedLinearIdentifier:
    """Weighted ridge regression for x_{k+1} = A x_k + B u_k + c + w_k."""

    def __init__(self, forgetting_factor: float, lambda_reg: float):
        self.forgetting_factor = forgetting_factor
        self.lambda_reg = lambda_reg
        self.A = None
        self.B = None
        self.c = None
        self.w_bound = None

    def fit(self, x_data: np.ndarray, u_data: np.ndarray, disturbance_quantile: float, disturbance_scale: float):
        n_samples = x_data.shape[0]
        if n_samples < 3:
            raise ValueError("Not enough samples for model identification.")

        Xk = x_data[:-1, :]
        Uk = u_data[:-1, :]
        Xkp1 = x_data[1:, :]

        Phi = np.hstack([Xk, Uk, np.ones((Xk.shape[0], 1))])

        # Exponential forgetting: newer samples receive larger weights.
        weights = self.forgetting_factor ** np.arange(Phi.shape[0] - 1, -1, -1)
        W = np.diag(weights)

        reg = self.lambda_reg * np.eye(Phi.shape[1])
        Theta = np.linalg.solve(Phi.T @ W @ Phi + reg, Phi.T @ W @ Xkp1)

        n_x = Xk.shape[1]
        self.A = Theta[:n_x, :].T
        self.B = Theta[n_x:-1, :].T
        self.c = Theta[-1, :].reshape(-1, 1)

        pred = (Phi @ Theta)
        residuals = Xkp1 - pred
        abs_q = np.quantile(np.abs(residuals), disturbance_quantile, axis=0)
        self.w_bound = disturbance_scale * abs_q

        return residuals


class RobustLinearMPC:
    """Robust MPC with disturbance tube-style constraint tightening."""

    def __init__(self, A, B, c, w_bound, umin, umax, xmin, xmax, cfg: MPCConfig):
        self.A = A
        self.B = B
        self.c = c.reshape(-1, 1)
        self.w_bound = w_bound.reshape(-1)
        self.umin = umin.reshape(-1)
        self.umax = umax.reshape(-1)
        self.xmin = xmin.reshape(-1)
        self.xmax = xmax.reshape(-1)
        self.cfg = cfg

        self.nx = A.shape[0]
        self.nu = B.shape[1]

        self.Q = np.diag(cfg.q_state_diag)
        self.R = np.diag(cfg.r_control_diag)

    def _compute_tightening_sequence(self):
        """r_{t+1} = |A| r_t + w_bound; tighten state constraints by r_t."""
        Aabs = np.abs(self.A)
        r = np.zeros(self.nx)
        rs = [r.copy()]
        for _ in range(self.cfg.h_p):
            r = Aabs @ r + self.w_bound
            rs.append(r.copy())
        return np.array(rs) * self.cfg.robust_tightening_scale

    def solve(self, x0, x_ref, u_ref):
        H_p = self.cfg.h_p
        H_c = self.cfg.h_c

        x = cp.Variable((self.nx, H_p + 1))
        u = cp.Variable((self.nu, H_c))
        s = cp.Variable((self.nx, H_p + 1), nonneg=True)

        tightening = self._compute_tightening_sequence()

        constraints = [x[:, 0] == x0.reshape(-1)]
        objective = 0

        for t in range(H_p):
            ut = u[:, min(t, H_c - 1)]
            constraints += [x[:, t + 1] == self.A @ x[:, t] + self.B @ ut + self.c.reshape(-1)]

            constraints += [ut <= self.umax, ut >= self.umin]

            x_upper = self.xmax - tightening[t + 1]
            x_lower = self.xmin + tightening[t + 1]
            constraints += [x[:, t + 1] <= x_upper + s[:, t + 1], x[:, t + 1] >= x_lower - s[:, t + 1]]

            objective += cp.quad_form(x[:, t + 1] - x_ref.reshape(-1), self.Q)
            objective += cp.quad_form(ut - u_ref.reshape(-1), self.R)
            objective += 1e5 * cp.sum(s[:, t + 1])

            if t > 0 and t < H_c:
                objective += 1e-2 * cp.sum_squares(u[:, t] - u[:, t - 1])

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if u.value is None:
            raise RuntimeError(f"MPC solve failed, status={prob.status}")

        return u.value[:, 0], prob.value


def detect_time_column(sample_df: pd.DataFrame):
    if "松散回潮-出料含水率时间" in sample_df.columns:
        return "松散回潮-出料含水率时间"
    if "松散回潮时间" in sample_df.columns:
        return "松散回潮时间"
    raise ValueError("Time column not found in worksheet.")


def parse_minutes_since_start(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    times = {}
    first_dt = None
    for idx, time_val in valid.items():
        try:
            dt = datetime.strptime(str(time_val), "%Y/%m/%d %H:%M:%S")
            if first_dt is None:
                first_dt = dt
            times[idx] = (dt - first_dt).total_seconds() / 60.0
        except (ValueError, TypeError):
            continue

    out = pd.Series(np.nan, index=series.index)
    for idx, mins in times.items():
        out.loc[idx] = mins
    return out


def load_and_preprocess(train_path: str, cfg: MPCConfig):
    print("Starting data loading and preprocessing...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    xls = pd.ExcelFile(train_path)
    sample_df = pd.read_excel(train_path, sheet_name=xls.sheet_names[0])
    time_col = detect_time_column(sample_df)

    all_batches = {}
    for i, sheet_name in enumerate(xls.sheet_names):
        df = pd.read_excel(train_path, sheet_name=sheet_name).copy()

        df[time_col] = parse_minutes_since_start(df[time_col])
        df = df.infer_objects(copy=False)
        df.interpolate(method="linear", limit_direction="both", inplace=True)
        df.dropna(subset=[time_col] + STATES_NAMES_CN + INPUTS_NAMES_CN, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < cfg.h_p + 2:
            continue

        all_batches[i] = {
            "name": sheet_name,
            "x": df[STATES_NAMES_CN].values,
            "u": df[INPUTS_NAMES_CN].values,
        }

    if not all_batches:
        raise ValueError("No valid batch was found after preprocessing.")

    x_all = np.vstack([data["x"] for data in all_batches.values()])
    u_all = np.vstack([data["u"] for data in all_batches.values()])

    print(f"Data loading finished. Loaded {len(all_batches)} valid batches.")
    return all_batches, x_all, u_all


def derive_input_bounds(u_all: np.ndarray, cfg: MPCConfig):
    u_min_hist = np.min(u_all, axis=0)
    u_max_hist = np.max(u_all, axis=0)

    spread = np.maximum(np.abs(u_max_hist - u_min_hist) * cfg.max_deviation_from_historical_percent,
                        np.array(cfg.min_abs_deviation_allowance))
    umin = u_min_hist - spread
    umax = u_max_hist + spread
    return umin, umax


def run_closed_loop_experiment(train_path: str, save_dir: str):
    cfg = MPCConfig()
    np.random.seed(cfg.random_seed)
    os.makedirs(save_dir, exist_ok=True)

    all_batches, x_all, u_all = load_and_preprocess(train_path, cfg)

    identifier = WeightedLinearIdentifier(
        forgetting_factor=cfg.forgetting_factor,
        lambda_reg=cfg.lambda_reg,
    )
    residuals = identifier.fit(
        x_data=x_all,
        u_data=u_all,
        disturbance_quantile=cfg.disturbance_quantile,
        disturbance_scale=cfg.disturbance_scale,
    )

    umin, umax = derive_input_bounds(u_all, cfg)
    xmin = np.quantile(x_all, 0.01, axis=0)
    xmax = np.quantile(x_all, 0.99, axis=0)

    x_ref = np.median(x_all, axis=0)
    u_ref = np.median(u_all, axis=0)

    mpc = RobustLinearMPC(
        A=identifier.A,
        B=identifier.B,
        c=identifier.c,
        w_bound=identifier.w_bound,
        umin=umin,
        umax=umax,
        xmin=xmin,
        xmax=xmax,
        cfg=cfg,
    )

    x_curr = all_batches[sorted(all_batches.keys())[0]]["x"][0].copy()
    xs = [x_curr.copy()]
    us = []
    costs = []
    solve_times_ms = []

    for _ in range(cfg.sim_steps):
        t0 = time.perf_counter()
        u_cmd, cost = mpc.solve(x_curr, x_ref, u_ref)
        solve_times_ms.append((time.perf_counter() - t0) * 1e3)

        # Closed-loop simulation using identified model + bounded random disturbance.
        w = np.random.uniform(-identifier.w_bound, identifier.w_bound)
        x_next = identifier.A @ x_curr + identifier.B @ u_cmd + identifier.c.reshape(-1) + w

        xs.append(x_next.copy())
        us.append(u_cmd.copy())
        costs.append(cost)
        x_curr = x_next

    xs = np.array(xs)
    us = np.array(us)
    costs = np.array(costs)

    tracking_rmse = np.sqrt(np.mean((xs[1:] - x_ref.reshape(1, -1)) ** 2, axis=0))

    np.savez(
        os.path.join(save_dir, "tobacco_robust_mpc_metrics.npz"),
        x_traj=xs,
        u_traj=us,
        costs=costs,
        solve_times_ms=np.array(solve_times_ms),
        x_ref=x_ref,
        u_ref=u_ref,
        tracking_rmse=tracking_rmse,
        disturbance_bound=identifier.w_bound,
        residuals=residuals,
    )

    # Plot summary.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    t_x = np.arange(xs.shape[0])
    for i, name in enumerate(STATES_NAMES_EN):
        axes[0, 0].plot(t_x, xs[:, i], label=name)
        axes[0, 0].axhline(x_ref[i], linestyle="--", alpha=0.7)
    axes[0, 0].set_title("State Trajectories")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    t_u = np.arange(us.shape[0])
    for i in range(min(4, us.shape[1])):
        axes[0, 1].plot(t_u, us[:, i], label=INPUTS_NAMES_EN[i])
    axes[0, 1].set_title("Input Trajectories (first 4 channels)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(costs)
    axes[1, 0].set_title("Stage Objective")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].grid(True)

    axes[1, 1].plot(solve_times_ms)
    axes[1, 1].set_title("Solve Time [ms]")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True)

    fig.savefig(os.path.join(save_dir, "tobacco_robust_mpc_summary.png"), dpi=160)
    plt.close(fig)

    print("Robust MPC run completed.")
    print(f"Tracking RMSE: {tracking_rmse}")
    print(f"Mean solve time [ms]: {np.mean(solve_times_ms):.3f}")
    print(f"Saved plot: {os.path.join(save_dir, 'tobacco_robust_mpc_summary.png')}")


def main():
    train_path = os.environ.get("TRAIN_XLS_PATH", "")
    if not train_path:
        print("Please provide your training Excel path by env variable TRAIN_XLS_PATH")
        print("Example:")
        print("  TRAIN_XLS_PATH='/path/to/train.xls' python experiments/tobacco_robust_mpc/run_tobacco_robust_mpc.py")
        sys.exit(1)

    run_closed_loop_experiment(
        train_path=train_path,
        save_dir="experiments/tobacco_robust_mpc/results",
    )


if __name__ == "__main__":
    main()
