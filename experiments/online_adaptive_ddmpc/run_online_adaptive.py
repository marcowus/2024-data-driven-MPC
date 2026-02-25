import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from casadi import DM, solve, vertcat, horzcat

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from four_tanks import A, B, C, D, n, m, p, tanks_dyn
from History_Data import collect_system_identification_data
from Hankel_matrix import construct_hankel_data_matrices
from Data_Driven_MPC import generate_DDMPC_robust_solver as build_controller
from system_id import shift_and_replace


class OnlineAdaptiveConfig:
    def __init__(self):
        # data and horizon
        self.data_window = 420
        self.initial_window_length = n
        self.forecast_horizon = 28
        self.actuation_steps = n

        # closed-loop length
        self.mpc_cycles = 80
        self.time_step = 1.0 / self.actuation_steps

        # online updates
        self.online_hankel_update_every = 3
        self.forgetting_factor = 0.992

        # noise settings
        self.measurement_noise_bound = 1.8e-3
        self.input_disturbance_enabled = True
        self.input_disturbance_level = 1.2
        self.output_disturbance_enabled = True
        self.output_disturbance_level = self.measurement_noise_bound

        # bounds/cost
        self.actuator_limits = [9.5] * m
        self.sensor_limits = [np.inf] * p
        self.output_penalty_matrix = 2.8 * np.eye(p)
        self.input_penalty_matrix = 1.2e-4 * np.eye(m)

        # solver
        self.use_constraint_regularization = True
        self.use_input_rate_penalty = True
        self.input_rate_weight = 1e-3
        self.solver_type = "osqp"
        self.verbose_solver = False
        self.compile_functions = True

        # robust penalties
        self.sigma_regularization = 1.2e3
        self.trajectory_regularization = 0.12 / self.measurement_noise_bound


def apply_column_forgetting(H_combined: np.ndarray, forgetting_factor: float) -> np.ndarray:
    """Scale Hankel columns with exponentially decaying weights (older columns receive lower weight)."""
    num_cols = H_combined.shape[1]
    col_weights = forgetting_factor ** np.arange(num_cols - 1, -1, -1)
    return H_combined * col_weights[np.newaxis, :]


def update_data_buffer(data: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
    """FIFO update for row-wise data buffer."""
    step = new_rows.shape[0]
    if step >= data.shape[0]:
        return new_rows[-data.shape[0] :, :].copy()
    out = np.empty_like(data)
    out[:-step, :] = data[step:, :]
    out[-step:, :] = new_rows
    return out


def run_experiment(save_dir: str = "experiments/online_adaptive_ddmpc/results"):
    os.makedirs(save_dir, exist_ok=True)
    cfg = OnlineAdaptiveConfig()

    rng = np.random.default_rng(2026)
    f_dis, g_dis, _, _, _ = tanks_dyn()

    # reference equilibrium
    steady_state_matrix = vertcat(horzcat(DM.eye(n) - A, -B), horzcat(C, D))
    target_conditions = DM([0, 0, 0, 0, 0.65, 0.77])
    steady_state_solution = solve(steady_state_matrix, target_conditions)
    reference_inputs = steady_state_solution[-m:].full()
    reference_outputs = steady_state_solution[:p].full()

    # start equilibrium
    init_conditions_rhs = DM(vertcat(DM.zeros(n), DM([0.4, 0.4])))
    init_steady_state = solve(steady_state_matrix, init_conditions_rhs)
    current_state = init_steady_state[:-m].full().squeeze()
    equilibrium_input = init_steady_state[-m:].full()

    # initial identification data
    historical_inputs, historical_outputs = collect_system_identification_data(
        m,
        p,
        cfg.data_window,
        f_dis,
        g_dis,
        np.zeros((n,)),
        cfg.input_disturbance_enabled,
        cfg.input_disturbance_level,
        cfg.output_disturbance_enabled,
        cfg.output_disturbance_level,
        None,
        np.zeros((cfg.data_window, n)),
        cfg.actuator_limits,
        cfg.time_step,
    )

    # build solver (dimensions fixed by data window)
    mpc_solver, substitute_parameters, extract_solution, extract_trajectories = build_controller(
        n,
        m,
        p,
        cfg.data_window,
        cfg.initial_window_length,
        cfg.forecast_horizon,
        cfg.use_constraint_regularization,
        cfg.use_input_rate_penalty,
        cfg.input_rate_weight,
        cfg.solver_type,
        cfg.verbose_solver,
        cfg.compile_functions,
    )

    # state holders
    input_window = np.zeros((m * cfg.initial_window_length, 1))
    output_window = np.zeros((p * cfg.initial_window_length, 1))

    g_guess = np.zeros((cfg.data_window - (cfg.initial_window_length + cfg.forecast_horizon) + 1, 1))
    sigma_guess = np.zeros((p * (cfg.forecast_horizon + cfg.initial_window_length), 1))
    input_traj_guess = np.zeros((m * (cfg.forecast_horizon + cfg.initial_window_length), 1))
    output_traj_guess = np.zeros((p * (cfg.forecast_horizon + cfg.initial_window_length), 1))

    total_steps = 1 + cfg.actuation_steps * cfg.mpc_cycles
    input_history = np.zeros((total_steps, m))
    output_history = np.zeros((total_steps, p))
    solver_time_ms = []
    pe_rank_log = []

    # collect initial window measurements
    noise = rng.uniform(
        -cfg.output_disturbance_level,
        cfg.output_disturbance_level,
        (p * cfg.initial_window_length, 1),
    )
    for step in range(cfg.initial_window_length):
        input_window[m * step : m * (step + 1)] = equilibrium_input
        measured_output = g_dis(current_state, input_window[m * step : m * (step + 1)]).full()
        output_window[p * step : p * (step + 1)] = measured_output + noise[p * step : p * (step + 1)]
        current_state = f_dis(current_state, input_window[m * step : m * (step + 1)]).full().squeeze()

    input_history[0] = input_window[-m:].squeeze()
    output_history[0] = output_window[-p:].squeeze()

    dual_x = None
    dual_g = None

    for iteration in range(0, cfg.actuation_steps * cfg.mpc_cycles, cfg.actuation_steps):
        # online Hankel rebuild (periodic)
        if (iteration // cfg.actuation_steps) % cfg.online_hankel_update_every == 0:
            H_combined, _, _, _, _ = construct_hankel_data_matrices(
                n,
                cfg.initial_window_length,
                cfg.forecast_horizon,
                historical_inputs,
                historical_outputs,
            )
            H_weighted = apply_column_forgetting(H_combined, cfg.forgetting_factor)
            pe_rank_log.append(np.linalg.matrix_rank(H_weighted[: m * (n + cfg.initial_window_length + cfg.forecast_horizon), :]))

        parameter_vector = vertcat(
            -DM(cfg.actuator_limits),
            DM(cfg.actuator_limits),
            -DM(cfg.sensor_limits),
            DM(cfg.sensor_limits),
            DM(reference_inputs),
            DM(reference_outputs),
            DM(cfg.output_penalty_matrix).reshape((-1, 1)),
            DM(cfg.input_penalty_matrix).reshape((-1, 1)),
            DM(cfg.trajectory_regularization),
            DM(cfg.measurement_noise_bound),
            DM(cfg.sigma_regularization),
            DM(input_window),
            DM(output_window),
            DM(H_weighted).reshape((-1, 1)),
            DM(g_guess),
            DM(sigma_guess),
            DM(input_traj_guess),
            DM(output_traj_guess),
        )

        x0, lbx, ubx, lbg, ubg = substitute_parameters(parameter_vector)
        if dual_x is None:
            dual_x = DM.zeros(x0.shape)
            dual_g = DM.zeros(lbg.shape)

        tic = time.perf_counter()
        sol = mpc_solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameter_vector, lam_x0=dual_x, lam_g0=dual_g)
        toc = time.perf_counter()
        solver_time_ms.append((toc - tic) * 1e3)

        opt_vars = sol["x"]
        dual_x = sol["lam_x"]
        dual_g = sol["lam_g"]

        g_guess, sigma_guess, input_traj_guess, output_traj_guess, _ = extract_solution(parameter_vector, opt_vars)
        input_traj_plot, _ = extract_trajectories(parameter_vector, opt_vars)

        next_inputs = input_traj_plot[cfg.initial_window_length : cfg.initial_window_length + cfg.actuation_steps, :].full()
        input_history[iteration + 1 : iteration + 1 + cfg.actuation_steps] = next_inputs.squeeze()

        step_noise = rng.uniform(-cfg.output_disturbance_level, cfg.output_disturbance_level, (cfg.actuation_steps, p))

        new_u_samples = []
        new_y_samples = []
        for act_step in range(cfg.actuation_steps):
            u_k = input_history[iteration + 1 + act_step]
            y_k = g_dis(current_state, u_k).full().squeeze() + step_noise[act_step]
            output_history[iteration + 1 + act_step] = y_k

            current_state = f_dis(current_state, u_k).full().squeeze()

            input_window = shift_and_replace(input_window, u_k)
            output_window = shift_and_replace(output_window, y_k)
            input_traj_guess = shift_and_replace(input_traj_guess, reference_inputs)
            output_traj_guess = shift_and_replace(output_traj_guess, reference_outputs)

            new_u_samples.append(u_k)
            new_y_samples.append(y_k)

        historical_inputs = update_data_buffer(historical_inputs, np.array(new_u_samples))
        historical_outputs = update_data_buffer(historical_outputs, np.array(new_y_samples))

    # save metrics
    tracking_rmse = np.sqrt(np.mean((output_history - reference_outputs.squeeze()) ** 2, axis=0))
    np.savez(
        os.path.join(save_dir, "metrics_online_adaptive.npz"),
        solver_time_ms=np.array(solver_time_ms),
        pe_rank=np.array(pe_rank_log),
        tracking_rmse=tracking_rmse,
        input_history=input_history,
        output_history=output_history,
    )

    # visualize
    t = np.linspace(0, cfg.actuation_steps * cfg.mpc_cycles * cfg.time_step, total_steps)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for j in range(m):
        axes[0, 0].plot(t, input_history[:, j], label=f"u{j+1}")
        axes[0, 0].plot(t, reference_inputs[j] * np.ones_like(t), "--", alpha=0.7)
    axes[0, 0].set_title("Inputs (online adaptive Hankel)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    for j in range(p):
        axes[0, 1].plot(t, output_history[:, j], label=f"y{j+1}")
        axes[0, 1].plot(t, reference_outputs[j] * np.ones_like(t), "--", alpha=0.7)
    axes[0, 1].set_title("Outputs tracking")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(solver_time_ms)
    axes[1, 0].set_title("Solver time per MPC iteration [ms]")
    axes[1, 0].grid(True)

    axes[1, 1].plot(pe_rank_log)
    axes[1, 1].set_title("PE proxy rank across online updates")
    axes[1, 1].grid(True)

    fig.savefig(os.path.join(save_dir, "online_adaptive_summary.png"), dpi=160)
    plt.close(fig)

    print("Online adaptive DDMPC completed.")
    print(f"Tracking RMSE: {tracking_rmse}")
    print(f"Mean solver time [ms]: {np.mean(solver_time_ms):.3f}")
    print(f"Saved plot: {os.path.join(save_dir, 'online_adaptive_summary.png')}")


if __name__ == "__main__":
    run_experiment()
