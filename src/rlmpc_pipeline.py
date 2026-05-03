"""High-level RLMPC/Gnu-RL pipeline contracts and offline imitation helpers."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RLMPCPaths:
    offline_checkpoint_dir: Path = Path("results/models/rlmpc/offline")
    offline_bundle_dir: Path = Path("results/models/rlmpc/bundles")
    raw_result_dir: Path = Path("results/raw/rlmpc")
    summary_dir: Path = Path("results/summaries/rlmpc")


@dataclass(frozen=True)
class RLMPCExperiment:
    controller: str
    scenario: str
    occupant_label: str | None = None
    tdyn_variant: str | None = None

    @property
    def result_slug(self) -> str:
        parts = [self.controller, self.scenario]
        if self.occupant_label:
            parts.append(self.occupant_label)
        if self.tdyn_variant:
            parts.append(self.tdyn_variant)
        return "__".join(parts)


def offline_checkpoint_path(experiment: RLMPCExperiment, paths: RLMPCPaths = RLMPCPaths()) -> Path:
    return paths.offline_checkpoint_dir / f"{experiment.result_slug}.pt"


def offline_bundle_path(experiment: RLMPCExperiment, paths: RLMPCPaths = RLMPCPaths()) -> Path:
    return paths.offline_bundle_dir / f"{experiment.result_slug}.json"


def raw_rollout_path(experiment: RLMPCExperiment, paths: RLMPCPaths = RLMPCPaths()) -> Path:
    return paths.raw_result_dir / experiment.scenario / f"{experiment.result_slug}.parquet"


def metrics_path(experiment: RLMPCExperiment, paths: RLMPCPaths = RLMPCPaths()) -> Path:
    return paths.summary_dir / experiment.scenario / f"{experiment.result_slug}.json"


DATASET_NAME = "annex96_ce1_vt_neighborhood"
CONTROL_BUILDING_NAME = "resstock-amy2018-2021-release-1-247942"
OCCUPANCY_BUILDING_NAME = "resstock-amy2018-2021-release-1-20199"
JAN_START = 0
JAN_END = 743
FEB_START = 744
FEB_END = 1415
WEEK_HOURS = 7 * 24

FITTED_OCCUPANT_PARAMS = {
    "occ1_tolerant": {"T_pref": 21.861014, "b": 1.198849, "max_prob": 0.4},
    "occ2_sensitive": {"T_pref": 23.035896, "b": 3.451184, "max_prob": 0.4},
    "occ3_cold": {"T_pref": 19.957358, "b": 2.618482, "max_prob": 0.4},
    "occ4_hot": {"T_pref": 23.909844, "b": 2.570147, "max_prob": 0.4},
}

NOTEBOOK_RLMPC_OBSERVATIONS = [
    "month",
    "hour",
    "outdoor_dry_bulb_temperature",
    "direct_solar_irradiance",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point",
    "comfort_band",
    "indoor_dry_bulb_temperature_cooling_delta",
    "indoor_dry_bulb_temperature_heating_delta",
    "hvac_mode",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "cooling_electricity_consumption",
    "heating_electricity_consumption",
    "dhw_electricity_consumption",
    "electrical_storage_electricity_consumption",
]


def set_experiment_seed(seed: int) -> int:
    """Seed numpy, torch, and Python random when available."""

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def observation_indices_from_names(names: list[str]) -> dict[str, Any]:
    """Resolve the observation columns used by the notebook RLMPC code."""

    required = [
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_heating_set_point",
        "outdoor_dry_bulb_temperature",
        "direct_solar_irradiance",
        "outdoor_dry_bulb_temperature_predicted_1",
        "outdoor_dry_bulb_temperature_predicted_2",
        "outdoor_dry_bulb_temperature_predicted_3",
        "direct_solar_irradiance_predicted_1",
        "direct_solar_irradiance_predicted_2",
        "direct_solar_irradiance_predicted_3",
    ]
    missing = [name for name in required if name not in names]
    if missing:
        available = "\n".join(f"  {i:>2}: {name}" for i, name in enumerate(names))
        raise RuntimeError(f"Missing RLMPC observations: {missing}\nAvailable:\n{available}")

    def idx(name: str) -> int:
        return names.index(name)

    return {
        "tin": idx("indoor_dry_bulb_temperature"),
        "tsp": idx("indoor_dry_bulb_temperature_heating_set_point"),
        "tmax": idx("indoor_dry_bulb_temperature_cooling_set_point")
        if "indoor_dry_bulb_temperature_cooling_set_point" in names
        else None,
        "hour": idx("hour") if "hour" in names else None,
        "tout": idx("outdoor_dry_bulb_temperature"),
        "solar": idx("direct_solar_irradiance"),
        "tout_p": [idx(f"outdoor_dry_bulb_temperature_predicted_{h}") for h in [1, 2, 3]],
        "solar_p": [idx(f"direct_solar_irradiance_predicted_{h}") for h in [1, 2, 3]],
    }


def observation_names_for_dataset(obs_dim: int, preset: str = "auto") -> tuple[str, list[str]]:
    """Return the observation layout for the saved January notebook dataset."""

    if preset == "notebook_mpc_28":
        return "notebook_mpc_28", NOTEBOOK_RLMPC_OBSERVATIONS
    if preset != "auto":
        raise ValueError("Only 'auto' and 'notebook_mpc_28' are supported for offline RLMPC data.")
    if obs_dim == len(NOTEBOOK_RLMPC_OBSERVATIONS):
        return "notebook_mpc_28", NOTEBOOK_RLMPC_OBSERVATIONS
    raise RuntimeError(
        f"Dataset has {obs_dim} observation columns. The migrated notebook path "
        f"currently expects {len(NOTEBOOK_RLMPC_OBSERVATIONS)} columns."
    )


class OccupantComfortReward:
    """Notebook reward wrapper: use override setpoint when an occupant wrapper supplies one."""

    def __new__(cls, *args, **kwargs):
        from citylearn.reward_function import ComfortReward

        class _OccupantComfortReward(ComfortReward):
            def calculate(self, observations):
                reward_list = []
                for observation in observations:
                    if "override_setpoint" in observation:
                        observation["indoor_dry_bulb_temperature_heating_set_point"] = observation[
                            "override_setpoint"
                        ]
                    reward_list.append(super().calculate([observation])[0])
                return [sum(reward_list)] if self.central_agent else reward_list

        return _OccupantComfortReward(*args, **kwargs)


class Occupant:
    def __init__(self, T_pref, b, max_prob=0.4, duration=3):
        self.T_pref = float(T_pref)
        self.b = float(b)
        self.max_prob = float(max_prob)
        self.duration = int(duration)
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def prob_increase(self, Tin):
        import numpy as np

        return min(0.002 * np.exp(self.b * max(0.0, self.T_pref - float(Tin))), self.max_prob)

    def prob_decrease(self, Tin):
        import numpy as np

        return min(0.002 * np.exp(self.b * max(0.0, float(Tin) - self.T_pref)), self.max_prob)

    @staticmethod
    def override_delta(p_override):
        if p_override < 0.2:
            return 0.5
        if p_override < 0.4:
            return 1.0
        return 2.0

    def clear_override(self):
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def step(self, Tin, scheduled_Tsp, occupied=True):
        import numpy as np

        self.last_override_started = False
        if not occupied:
            self.clear_override()
            return float(scheduled_Tsp)

        if self.override_timer > 0:
            self.override_timer -= 1
            return float(self.override_setpoint)

        p_inc = self.prob_increase(Tin)
        p_dec = self.prob_decrease(Tin)
        r = np.random.rand()

        if r < p_inc:
            new_sp = float(scheduled_Tsp) + self.override_delta(p_inc)
        elif r < p_inc + p_dec:
            new_sp = float(scheduled_Tsp) - self.override_delta(p_dec)
        else:
            self.override_setpoint = None
            return float(scheduled_Tsp)

        self.override_setpoint = float(new_sp)
        self.override_timer = max(self.duration - 1, 0)
        self.last_override_started = True
        return float(new_sp)


def fitted_occupants() -> dict[str, Occupant]:
    return {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}


class OccupantWrapperWithFeedback:
    """Thermostat override wrapper with present-only intervention and feedback direction."""

    def __new__(cls, *args, **kwargs):
        import gymnasium as gym

        class _OccupantWrapperWithFeedback(gym.Wrapper):
            def __init__(self, env, occupant):
                super().__init__(env)
                self.occupant = occupant
                if hasattr(env, "observation_names"):
                    self.observation_names = env.observation_names
                if hasattr(env.unwrapped, "action_names"):
                    self.action_names = env.unwrapped.action_names
                self.observation_space = env.observation_space
                self.action_space = env.action_space

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.occupant.override_timer = 0
                self.occupant.override_setpoint = None
                self.occupant.last_override_started = False
                self.override_count = 0
                self.last_feedback = 0
                b = self.env.unwrapped.buildings[0]
                self.effective_setpoints = [
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                ]
                self.baseline_setpoints = [
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                ]
                return obs, info

            def step(self, action):
                b = self.env.unwrapped.buildings[0]
                current_ix = b.time_step
                Tin = float(b.energy_simulation.indoor_dry_bulb_temperature[current_ix])
                next_ix = min(
                    current_ix + 1,
                    len(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point) - 1,
                )
                occupied_next = float(b.energy_simulation.occupant_count[next_ix]) > 0.0
                scheduled_next_Tsp = float(
                    b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[next_ix]
                )
                new_Tsp = self.occupant.step(Tin, scheduled_next_Tsp, occupied=occupied_next)

                self.last_feedback = 0
                if self.occupant.last_override_started:
                    self.override_count += 1
                    if new_Tsp > scheduled_next_Tsp:
                        self.last_feedback = +1
                    elif new_Tsp < scheduled_next_Tsp:
                        self.last_feedback = -1

                b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[next_ix] = new_Tsp
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.effective_setpoints.append(
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                )
                self.baseline_setpoints.append(
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                )
                return obs, reward, terminated, truncated, info

        return _OccupantWrapperWithFeedback(*args, **kwargs)


def _dataset_paths() -> tuple[Path, Path]:
    dataset_dir = Path("data") / "datasets" / DATASET_NAME
    return dataset_dir / "schema.json", dataset_dir


def load_building_occupancy_series(building_name: str = OCCUPANCY_BUILDING_NAME):
    import pandas as pd

    schema_path, dataset_dir = _dataset_paths()
    schema = json.loads(schema_path.read_text())
    csv_path = dataset_dir / schema["buildings"][building_name]["energy_simulation"]
    return pd.read_csv(csv_path)["occupant_count"].astype(int).to_numpy()


def fit_bspline_occupancy_model(*, seed: int, building_name: str = OCCUPANCY_BUILDING_NAME):
    from src.occupants.bspline_preferences import FixedBSplineIMCOccupancyPredictor

    occupancy_series = load_building_occupancy_series(building_name)
    occupancy_model = FixedBSplineIMCOccupancyPredictor(
        period=24,
        spline_degree=3,
        n_internal_knots=4,
        l2_phi=1e-3,
        min_samples_per_state=25,
        maxiter=1000,
        random_state=int(seed),
    )
    occupancy_model.fit(occupancy_series)
    return occupancy_model, occupancy_series


def forecast_occupancy_from_bsplines(occupancy_model, current_occupancy: int, time_step: int, horizon: int, alpha=0.5):
    import numpy as np

    occ_hat_h, _ = occupancy_model.predict_expected_occupancy(
        current_state=int(current_occupancy),
        start_t=int(time_step),
        horizon=int(horizon),
    )
    occ_bin_h = (occ_hat_h > float(alpha)).astype(float)
    return np.asarray(occ_hat_h, dtype=float), np.asarray(occ_bin_h, dtype=float)


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def _normalization_from_training(states, train_ids, indices) -> dict[str, float]:
    x_mean = states[train_ids, indices["tin"]].mean()
    x_std = states[train_ids, indices["tin"]].std() + 1e-6
    d1_mean = states[train_ids, indices["tout"]].mean()
    d1_std = states[train_ids, indices["tout"]].std() + 1e-6
    d2_mean = states[train_ids, indices["solar"]].mean()
    d2_std = states[train_ids, indices["solar"]].std() + 1e-6
    return _float_dict(
        {
            "x_mean": x_mean.detach().cpu(),
            "x_std": x_std.detach().cpu(),
            "d1_mean": d1_mean.detach().cpu(),
            "d1_std": d1_std.detach().cpu(),
            "d2_mean": d2_mean.detach().cpu(),
            "d2_std": d2_std.detach().cpu(),
        }
    )


def norm_x(value, normalization: dict[str, float]):
    return (value - normalization["x_mean"]) / normalization["x_std"]


def norm_sp(value, normalization: dict[str, float]):
    return norm_x(value, normalization)


def norm_d(value, normalization: dict[str, float]):
    import torch

    d_out = (value[..., 0:1] - normalization["d1_mean"]) / normalization["d1_std"]
    d_sol = (value[..., 1:2] - normalization["d2_mean"]) / normalization["d2_std"]
    return torch.cat([d_out, d_sol], dim=-1)


def _d_forecast_from_dataset(states, t: int, horizon: int, indices: dict[str, Any]):
    import torch

    n_samples = states.shape[0]
    end = min(t + horizon, n_samples)
    d_seq = torch.stack([states[t:end, indices["tout"]], states[t:end, indices["solar"]]], dim=1)
    if d_seq.shape[0] < horizon:
        d_seq = torch.cat([d_seq, d_seq[-1:].repeat(horizon - d_seq.shape[0], 1)], dim=0)
    return d_seq


def _sp_forecast_from_dataset(states, t: int, horizon: int, indices: dict[str, Any]):
    import torch

    n_samples = states.shape[0]
    end = min(t + horizon, n_samples)
    sp_seq = states[t:end, indices["tsp"]].view(-1, 1)
    if sp_seq.shape[0] < horizon:
        sp_seq = torch.cat([sp_seq, sp_seq[-1:].repeat(horizon - sp_seq.shape[0], 1)], dim=0)
    return sp_seq


def one_step_pred(mpc, x_t, u_t, d_seq):
    """Notebook one-step prediction loss, including its normalized-disturbance convention."""

    import torch

    C = mpc._pos(mpc.C_raw)
    Rm = mpc._pos(mpc.Rm_raw)
    Rout = mpc._pos(mpc.Rout_raw)
    Aeff = mpc._pos(mpc.Aeff_raw)
    Pnom = mpc._pos(mpc.Pnom_raw)

    Ac = -(1.0 / (Rm * C) + 1.0 / (Rout * C))
    A_disc = torch.exp(Ac * mpc.dt)
    eps = 1e-8
    phi = torch.where(torch.abs(Ac) > eps, (A_disc - 1.0) / Ac, torch.full_like(Ac, mpc.dt))

    Tout0 = d_seq[:, 0, 0]
    Isol0 = d_seq[:, 0, 1]
    cop0 = torch.clamp(mpc.cop_a + mpc.cop_b * Tout0, mpc.cop_min, mpc.cop_max)

    Bu0 = phi * (cop0 * Pnom / C)
    Bd0 = phi * (1.0 / (Rm * C))
    Bd1 = phi * (1.0 / (Rout * C))
    Bd2 = phi * (Aeff / C)

    return (
        A_disc * x_t.squeeze()
        + Bu0 * u_t.squeeze()
        + Bd0 * mpc.Tm
        + Bd1 * Tout0
        + Bd2 * Isol0
    ).view(1, 1)


def _state_dict_cpu(mpc) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in mpc.state_dict().items()}


def train_offline_diffmpc(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    horizon: int = 6,
    eta: float = 10.0,
    rho_u: float = 1.0,
    dt: float = 1.0,
    u_min: float = 0.0,
    u_max: float = 1.0,
    q_reg: float = 1e-4,
    qp_max_iter: int = 200,
    qp_eps: float = 1e-6,
    epochs: int = 200,
    lr: float = 1e-2,
    lambda_dyn: float = 4.0,
    batch_size: int = 32,
    patience: int = 7,
    seed: int = 0,
    device: str = "auto",
    max_samples: int | None = None,
    observation_preset: str = "auto",
) -> dict[str, Any]:
    """Run the notebook offline imitation learning loop and save checkpoint + bundle."""

    import numpy as np
    import torch

    from src.controllers.rlmpc import PaperDiffMPC, diffmpc_param_snapshot

    set_experiment_seed(seed)
    torch_device = resolve_device(device)

    data = np.load(dataset_path)
    states_np = data["states"][:, 0, :].astype(np.float32)
    actions_np = data["actions"][:, 0, :].astype(np.float32)
    next_states_np = data["next_states"][:, 0, :].astype(np.float32)
    if max_samples is not None:
        states_np = states_np[: int(max_samples)]
        actions_np = actions_np[: int(max_samples)]
        next_states_np = next_states_np[: int(max_samples)]

    if states_np.shape[0] < 4:
        raise RuntimeError("Need at least four samples for train/validation split.")

    observation_source, observation_names = observation_names_for_dataset(states_np.shape[-1], observation_preset)
    indices = observation_indices_from_names(observation_names)

    states = torch.tensor(states_np, device=torch_device)
    actions = torch.tensor(actions_np, device=torch_device)
    next_states = torch.tensor(next_states_np, device=torch_device)

    n_samples = int(states.shape[0])
    split = max(1, min(n_samples - 1, int(0.8 * n_samples)))
    train_ids = np.arange(0, split)
    val_ids = np.arange(split, n_samples)

    normalization = _normalization_from_training(states, train_ids, indices)

    def d_forecast(t: int):
        return _d_forecast_from_dataset(states, int(t), horizon, indices)

    def sp_forecast(t: int):
        return _sp_forecast_from_dataset(states, int(t), horizon, indices)

    mpc = PaperDiffMPC(
        T=horizon,
        eta=eta,
        rho_u=rho_u,
        dt=dt,
        u_min=u_min,
        u_max=u_max,
        q_reg=q_reg,
        qp_max_iter=qp_max_iter,
        qp_eps=qp_eps,
    ).to(torch_device)
    opt = torch.optim.Adam(mpc.parameters(), lr=lr)

    @torch.no_grad()
    def eval_loss(ids) -> float:
        mpc.eval()
        losses = []
        for t in ids:
            x_t = norm_x(states[t, indices["tin"]], normalization).view(1, 1)
            x_next_true = norm_x(next_states[t, indices["tin"]], normalization).view(1, 1)
            u_t = actions[t].view(1, 1)
            d_seq = norm_d(d_forecast(int(t)), normalization).view(1, horizon, 2)
            sp_seq = norm_sp(sp_forecast(int(t)), normalization).view(1, horizon, 1)
            u_mpc, _, _ = mpc(x_t, d_seq, sp_seq)
            x_next_pred = one_step_pred(mpc, x_t, u_t, d_seq)
            loss = lambda_dyn * (x_next_true - x_next_pred).pow(2).mean() + (u_t - u_mpc).pow(2).mean()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    best_val = float("inf")
    best_sd = None
    bad_epochs = 0
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []

    print(f"Training samples: {n_samples}")
    print(f"Observation source: {observation_source}")
    print(
        "Resolved indices: "
        f"Tin={indices['tin']} Tsp={indices['tsp']} Tout={indices['tout']} Solar={indices['solar']}"
    )
    print(f"Device: {torch_device}")

    for epoch in range(int(epochs)):
        mpc.train()
        rng.shuffle(train_ids)
        batch_losses = []

        for start in range(0, len(train_ids), int(batch_size)):
            batch = train_ids[start : start + int(batch_size)]
            opt.zero_grad()
            loss_batch = 0.0

            for t in batch:
                x_t = norm_x(states[t, indices["tin"]], normalization).view(1, 1)
                x_next_true = norm_x(next_states[t, indices["tin"]], normalization).view(1, 1)
                u_t = actions[t].view(1, 1)
                d_seq = norm_d(d_forecast(int(t)), normalization).view(1, horizon, 2)
                sp_seq = norm_sp(sp_forecast(int(t)), normalization).view(1, horizon, 1)

                u_mpc, _, _ = mpc(x_t, d_seq, sp_seq)
                x_next_pred = one_step_pred(mpc, x_t, u_t, d_seq)
                loss = lambda_dyn * (x_next_true - x_next_pred).pow(2).mean() + (u_t - u_mpc).pow(2).mean()
                loss_batch = loss_batch + loss

            loss_batch = loss_batch / max(1, len(batch))
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(mpc.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss_batch.item()))

        val = eval_loss(val_ids)
        train = float(np.mean(batch_losses))
        params = diffmpc_param_snapshot(mpc)
        history.append({"epoch": epoch, "train_loss": train, "val_loss": val, **params})
        print(
            f"Epoch {epoch:03d} | Train {train:.6f} | Val {val:.6f} | "
            f"C={params['C']:.3g} Rm={params['Rm']:.3g} Rout={params['Rout']:.3g} "
            f"Tm={params['Tm']:+.3f} Aeff={params['Aeff']:.3g} Pnom={params['Pnom']:.3g} "
            f"cop_a={params['cop_a']:.3g} cop_b={params['cop_b']:.3g}"
        )

        if val < best_val - 1e-6:
            best_val = val
            best_sd = _state_dict_cpu(mpc)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                print("Early stopping.")
                break

    if best_sd is None:
        raise RuntimeError("Offline training did not produce a checkpoint.")

    mpc.load_state_dict(best_sd)
    final_params = diffmpc_param_snapshot(mpc)

    model_config = {
        "horizon": int(horizon),
        "eta": float(eta),
        "rho_u": float(rho_u),
        "dt": float(dt),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "q_reg": float(q_reg),
        "qp_max_iter": int(qp_max_iter),
        "qp_eps": float(qp_eps),
    }
    training_config = {
        "epochs": int(epochs),
        "lr": float(lr),
        "lambda_dyn": float(lambda_dyn),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "seed": int(seed),
        "max_samples": None if max_samples is None else int(max_samples),
    }

    checkpoint = {
        "state_dict": best_sd,
        "best_val": float(best_val),
        "model_config": model_config,
        "training_config": training_config,
        "normalization": normalization,
        "indices": indices,
        "observation_source": observation_source,
        "observation_names": observation_names,
        "dataset_path": str(dataset_path),
        "history": history,
        "final_params": final_params,
    }

    checkpoint_path = Path(checkpoint_path)
    bundle_path = Path(bundle_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    bundle = deepcopy(checkpoint)
    bundle.pop("state_dict")
    bundle["checkpoint_path"] = str(checkpoint_path)
    bundle_path.write_text(json.dumps(bundle, indent=2))

    print(f"Best val: {best_val:.6f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved bundle: {bundle_path}")
    return checkpoint


def load_offline_checkpoint(checkpoint_path: str | Path, *, device: str = "auto"):
    import torch

    from src.controllers.rlmpc import PaperDiffMPC

    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model_config = checkpoint["model_config"]
    mpc = PaperDiffMPC(
        T=model_config["horizon"],
        eta=model_config["eta"],
        rho_u=model_config["rho_u"],
        dt=model_config["dt"],
        u_min=model_config["u_min"],
        u_max=model_config["u_max"],
        q_reg=model_config["q_reg"],
        qp_max_iter=model_config["qp_max_iter"],
        qp_eps=model_config["qp_eps"],
    ).to(torch_device)
    mpc.load_state_dict(checkpoint["state_dict"], strict=False)
    mpc.eval()
    return mpc, checkpoint, torch_device


class DiffMPCPolicyRaw:
    """Raw CityLearn observation policy, matching the notebook rollout code."""

    def __init__(self, mpc, normalization: dict[str, float], indices: dict[str, Any], *, device):
        self.mpc = mpc
        self.normalization = normalization
        self.indices = indices
        self.device = device
        self.w = 0.0
        self.gamma = 0.2
        self.last_history_bias = 0.0
        self.prev_x = None
        self.prev_u = None
        self.prev_d = None

    def reset(self):
        self.w = 0.0
        self.prev_x = None
        self.prev_u = None
        self.prev_d = None
        self.last_history_bias = 0.0

    def _norm_d(self, value):
        return norm_d(value, self.normalization)

    def predict(self, obs):
        import numpy as np
        import torch

        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.indices
        Tin = float(o[idx["tin"]])
        Tsp = float(o[idx["tsp"]])
        x = (Tin - self.normalization["x_mean"]) / self.normalization["x_std"]

        d_phys = torch.tensor([[o[idx["tout"]], o[idx["solar"]]]], device=self.device)
        d0 = self._norm_d(d_phys)

        if self.prev_x is not None:
            C = float(self.mpc._pos(self.mpc.C_raw))
            Rm = float(self.mpc._pos(self.mpc.Rm_raw))
            Rout = float(self.mpc._pos(self.mpc.Rout_raw))
            Aeff = float(self.mpc._pos(self.mpc.Aeff_raw))
            Pnom = float(self.mpc._pos(self.mpc.Pnom_raw))

            Ac = -(1 / (Rm * C) + 1 / (Rout * C))
            A = np.exp(Ac * self.mpc.dt)
            phi = (A - 1.0) / Ac if abs(Ac) > 1e-8 else self.mpc.dt

            tout_prev = float(self.prev_d[0, 0])
            isol_prev = float(self.prev_d[0, 1])
            cop = float(self.mpc.cop_a + self.mpc.cop_b * tout_prev)
            cop = np.clip(cop, self.mpc.cop_min, self.mpc.cop_max)

            Bu = phi * (cop * Pnom / C)
            Bd0 = phi * (1 / (Rm * C))
            Bd1 = phi * (1 / (Rout * C))
            Bd2 = phi * (Aeff / C)
            x_pred = (
                A * self.prev_x
                + Bu * self.prev_u
                + Bd0 * float(self.mpc.Tm.item())
                + Bd1 * tout_prev
                + Bd2 * isol_prev
                + self.w
            )
            self.w += self.gamma * (x - x_pred)

        x_t = torch.tensor([[x]], dtype=torch.float32, device=self.device)
        tout = [o[idx["tout"]], *[o[j] for j in idx["tout_p"]]]
        sol = [o[idx["solar"]], *[o[j] for j in idx["solar_p"]]]
        tout_h = [tout[min(i, 3)] for i in range(self.mpc.T)]
        sol_h = [sol[min(i, 3)] for i in range(self.mpc.T)]
        d_phys = torch.tensor(np.stack([tout_h, sol_h], axis=1), dtype=torch.float32, device=self.device)
        d_seq = self._norm_d(d_phys).view(1, self.mpc.T, 2)

        sp_now = (Tsp - self.normalization["x_mean"]) / self.normalization["x_std"]
        sp_seq = torch.full((1, self.mpc.T, 1), sp_now, dtype=torch.float32, device=self.device)
        w_seq = torch.full((1, self.mpc.T, 1), float(self.w), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u0, _, _ = self.mpc(x_t, d_seq, sp_seq, w_seq)

        u = float(u0.item())
        self.prev_x = x
        self.prev_u = u
        self.prev_d = d0
        self.last_history_bias = 0.0
        return [np.array([u], dtype=np.float32)]


def make_january_raw_env(*, seed: int = 49, reward_function=None):
    from src.envs import EnvConfig, make_citylearn_env

    dataset_dir = Path("data") / "datasets" / DATASET_NAME
    config = EnvConfig(
        schema_path=dataset_dir / "schema.json",
        dataset_dir=dataset_dir,
        buildings=[CONTROL_BUILDING_NAME],
        start_time_step=JAN_START,
        end_time_step=JAN_END,
        random_seed=int(seed),
        central_agent=True,
        active_actions=("heating_device",),
    )
    overrides = {}
    if reward_function is not None:
        overrides["reward_function"] = reward_function
    return make_citylearn_env(config, **overrides)


def run_january_raw_rollout(
    *,
    checkpoint_path: str | Path,
    seed: int = 49,
    reward_function=None,
    device: str = "auto",
    max_steps: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    env = make_january_raw_env(seed=seed, reward_function=reward_function)
    obs, _ = env.reset(seed=seed)
    env_indices = observation_indices_from_names(list(env.unwrapped.buildings[0].active_observations))
    policy = DiffMPCPolicyRaw(mpc, checkpoint["normalization"], env_indices, device=torch_device)

    terminated = False
    truncated = False
    reward_sum = 0.0
    actions = []
    step_count = 0
    while not (terminated or truncated):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_value = float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward)
        reward_sum += reward_value
        actions.append(float(action[0][0]))
        step_count += 1
        if max_steps is not None and step_count >= int(max_steps):
            break

    print(f"January reward (RAW) pretrained Diff-MPC: {reward_sum:.6f}")
    return {
        "env": env,
        "reward_sum": reward_sum,
        "actions": np.asarray(actions, dtype=float),
        "steps": step_count,
    }


def build_baseline_features(
    obs,
    policy_state: dict[str, Any] | None = None,
    *,
    indices: dict[str, Any],
    base_dim: int,
) -> Any:
    """Feature builder for the January ridge value baseline, matching the notebook."""

    import numpy as np

    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    Tin = o[indices["tin"]]
    Tsp = o[indices["tsp"]]
    Tout = o[indices["tout"]]
    Sol = o[indices["solar"]]
    hour = o[indices["hour"]] if indices.get("hour") is not None else 0.0

    if o.shape[0] >= base_dim + 5:
        occupied_now, feedback_now, feedback_ema, override_rate_ema, steps_since_override = o[base_dim : base_dim + 5]
    else:
        occupied_now = 0.0
        feedback_now = 0.0
        feedback_ema = 0.0
        override_rate_ema = 0.0
        steps_since_override = 24.0

    policy_state = {} if policy_state is None else dict(policy_state)
    if "history_bias" in policy_state:
        history_bias = float(policy_state["history_bias"])
    else:
        recent_signal = 0.75 * float(feedback_now) + 0.75 * float(feedback_ema)
        magnitude = 1.0 + 1.00 * float(override_rate_ema)
        decay = np.exp(-min(float(steps_since_override), 24.0) / 6.0)
        history_bias = float(np.clip(float(occupied_now) * recent_signal * magnitude * decay, -1.5, 1.5))

    w_est = float(policy_state.get("decision_w_est", policy_state.get("w_est", 0.0)))
    prev_u_raw = policy_state.get("prev_u", None)
    prev_u = 0.0 if prev_u_raw is None else float(prev_u_raw)
    has_prev_u = 0.0 if prev_u_raw is None else 1.0

    sin_h = np.sin(2 * np.pi * hour / 24.0)
    cos_h = np.cos(2 * np.pi * hour / 24.0)
    steps_since_override_norm = min(float(steps_since_override), 24.0) / 24.0

    return np.array(
        [
            Tin,
            Tsp,
            Tout,
            Sol,
            sin_h,
            cos_h,
            float(occupied_now),
            float(feedback_now),
            float(feedback_ema),
            float(override_rate_ema),
            steps_since_override_norm,
            float(history_bias),
            float(w_est),
            float(prev_u),
            float(has_prev_u),
        ],
        dtype=np.float32,
    )


def fit_january_value_baseline(
    *,
    dataset_path: str | Path,
    indices: dict[str, Any],
    base_dim: int,
    gamma: float = 0.99,
    ridge_alpha: float = 1e-3,
):
    import numpy as np
    from sklearn.linear_model import Ridge

    data = np.load(dataset_path)
    states_np = data["states"][:, 0, :].astype(np.float32)
    rewards_np = data["rewards"][:, 0].astype(np.float32)
    features = np.vstack(
        [build_baseline_features(states_np[t], indices=indices, base_dim=base_dim) for t in range(states_np.shape[0])]
    )

    returns = np.zeros_like(rewards_np)
    running = 0.0
    for t in reversed(range(states_np.shape[0])):
        running = float(rewards_np[t]) + float(gamma) * running
        returns[t] = running

    baseline_model = Ridge(alpha=float(ridge_alpha))
    baseline_model.fit(features, returns)
    return baseline_model


def compute_advantages_with_baseline(
    trajectory: list[dict[str, Any]],
    baseline_model,
    *,
    indices: dict[str, Any],
    base_dim: int,
    gamma: float = 0.99,
) -> tuple[Any, Any]:
    import numpy as np

    rewards = np.asarray([step["reward"] for step in trajectory], dtype=np.float32)
    returns = np.zeros(len(trajectory), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(trajectory))):
        running = float(rewards[t]) + float(gamma) * running
        returns[t] = running

    values = np.zeros(len(trajectory), dtype=np.float32)
    for t, step in enumerate(trajectory):
        state = step.get("baseline_state", step.get("policy_state"))
        z = build_baseline_features(step["obs"], state, indices=indices, base_dim=base_dim)
        values[t] = float(baseline_model.predict(z.reshape(1, -1))[0])

    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def d_forecast_from_obs(obs, horizon: int, *, indices: dict[str, Any], device):
    import numpy as np
    import torch

    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    tout_preds = [o[indices["tout"]]]
    sol_preds = [o[indices["solar"]]]
    for idx in indices["tout_p"]:
        if idx is not None:
            tout_preds.append(o[idx])
    for idx in indices["solar_p"]:
        if idx is not None:
            sol_preds.append(o[idx])

    tout_h = [tout_preds[k] if k < len(tout_preds) else tout_preds[-1] for k in range(int(horizon))]
    sol_h = [sol_preds[k] if k < len(sol_preds) else sol_preds[-1] for k in range(int(horizon))]
    return torch.tensor(np.stack([tout_h, sol_h], axis=1), dtype=torch.float32, device=device)


class StochasticDiffMPCPolicy:
    """Notebook PPO policy: Gaussian exploration around the DiffMPC mean action."""

    def __init__(self, mpc, normalization: dict[str, float], indices: dict[str, Any], *, sigma: float, device):
        self.mpc = mpc
        self.normalization = normalization
        self.indices = indices
        self.sigma = float(sigma)
        self.device = device
        self.w_est = 0.0
        self.gamma = 0.2
        self.prev_T = None
        self.prev_u = None
        self.prev_Tout = None
        self.prev_Isol = None
        self.last_history_bias = 0.0

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "w_est": float(self.w_est),
            "prev_T": None if self.prev_T is None else float(self.prev_T),
            "prev_u": None if self.prev_u is None else float(self.prev_u),
            "prev_Tout": None if self.prev_Tout is None else float(self.prev_Tout),
            "prev_Isol": None if self.prev_Isol is None else float(self.prev_Isol),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.w_est = float(state["w_est"])
        self.prev_T = state["prev_T"]
        self.prev_u = state["prev_u"]
        self.prev_Tout = state["prev_Tout"]
        self.prev_Isol = state["prev_Isol"]

    def reset(self) -> None:
        self.restore_state(
            {
                "w_est": 0.0,
                "prev_T": None,
                "prev_u": None,
                "prev_Tout": None,
                "prev_Isol": None,
            }
        )
        self.last_history_bias = 0.0

    def _updated_offset_state(self, Tin: float, Tout: float, Isol: float, state: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        local_state = {
            "w_est": float(state["w_est"]),
            "prev_T": state["prev_T"],
            "prev_u": state["prev_u"],
            "prev_Tout": state["prev_Tout"],
            "prev_Isol": state["prev_Isol"],
        }
        if local_state["prev_T"] is None:
            return local_state

        cop = float(self.mpc.cop_a + self.mpc.cop_b * local_state["prev_Tout"])
        cop = np.clip(cop, self.mpc.cop_min, self.mpc.cop_max)
        C = float(self.mpc._pos(self.mpc.C_raw))
        Rm = float(self.mpc._pos(self.mpc.Rm_raw))
        Rout = float(self.mpc._pos(self.mpc.Rout_raw))
        Aeff = float(self.mpc._pos(self.mpc.Aeff_raw))
        Pnom = float(self.mpc._pos(self.mpc.Pnom_raw))

        Ac = -(1 / (Rm * C) + 1 / (Rout * C))
        A = np.exp(Ac)
        phi = (A - 1) / Ac if abs(Ac) > 1e-6 else 1.0
        Bu = phi * (cop * Pnom / C)
        Bd0 = phi * (1 / (Rm * C))
        Bd1 = phi * (1 / (Rout * C))
        Bd2 = phi * (Aeff / C)
        T_pred = (
            A * local_state["prev_T"]
            + Bu * local_state["prev_u"]
            + Bd0 * self.mpc.Tm.item()
            + Bd1 * local_state["prev_Tout"]
            + Bd2 * local_state["prev_Isol"]
            + local_state["w_est"]
        )
        local_state["w_est"] = float(local_state["w_est"] + self.gamma * (Tin - T_pred))
        return local_state

    def _mpc_mean_action(self, obs, policy_state=None, update_internal: bool = True):
        import numpy as np
        import torch

        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.indices
        Tin = float(o[idx["tin"]])
        Tsp = float(o[idx["tsp"]])
        Tout = float(o[idx["tout"]])
        Isol = float(o[idx["solar"]])
        state_in = self.snapshot_state() if policy_state is None else policy_state
        state_work = self._updated_offset_state(Tin, Tout, Isol, state_in)

        x_t = torch.tensor(
            [[(Tin - self.normalization["x_mean"]) / self.normalization["x_std"]]],
            dtype=torch.float32,
            device=self.device,
        )
        d_phys = d_forecast_from_obs(o, self.mpc.T, indices=idx, device=self.device)
        d_seq = norm_d(d_phys, self.normalization).view(1, self.mpc.T, 2)
        sp_now = (Tsp - self.normalization["x_mean"]) / self.normalization["x_std"]
        sp_seq = torch.full((1, self.mpc.T, 1), sp_now, dtype=torch.float32, device=self.device)
        w_seq = torch.full((1, self.mpc.T, 1), float(state_work["w_est"]), dtype=torch.float32, device=self.device)
        u0, _, _ = self.mpc(x_t, d_seq, sp_seq, w_seq)
        mu = torch.clamp(u0.squeeze(), 0.0, 1.0)
        next_state = {
            "w_est": float(state_work["w_est"]),
            "prev_T": Tin,
            "prev_u": float(mu.item()),
            "prev_Tout": Tout,
            "prev_Isol": Isol,
        }
        if update_internal:
            self.restore_state(next_state)
        return mu, next_state

    def act(self, obs):
        import numpy as np
        import torch
        from torch.distributions import Normal

        policy_state = self.snapshot_state()
        mu, next_state = self._mpc_mean_action(obs, policy_state=policy_state, update_internal=False)
        self.restore_state(next_state)
        dist = Normal(mu, self.sigma)
        a_sample = dist.sample()
        logp = dist.log_prob(a_sample)
        a_env = torch.clamp(a_sample, 0.0, 1.0)
        action_env = [np.array([float(a_env.item())], dtype=np.float32)]
        return action_env, float(logp.item()), float(a_sample.item()), float(mu.item()), policy_state

    def log_prob(self, obs, a_sample_tensor, policy_state=None):
        from torch.distributions import Normal

        mu, _ = self._mpc_mean_action(obs, policy_state=policy_state, update_internal=False)
        dist = Normal(mu, self.sigma)
        return dist.log_prob(a_sample_tensor), mu


class StochasticDiffMPCPolicyTDyn(StochasticDiffMPCPolicy):
    """Stochastic DiffMPC policy using persistent T_dyn and optional B-spline occupancy gating."""

    def __init__(
        self,
        mpc,
        normalization: dict[str, float],
        indices: dict[str, Any],
        *,
        sigma: float,
        device,
        occupant,
        occupancy_model=None,
        occupancy_series=None,
        alpha_occ: float = 0.5,
        use_bspline_gate: bool = True,
        sim_start: int = FEB_START,
        T_dyn_init_mode: str = "schedule",
        delta_up: float = 0.5,
        delta_down: float = 0.5,
        drift_to_pref: float = 0.0,
        T_dyn_min: float = 18.0,
        T_dyn_max: float = 26.0,
    ):
        super().__init__(mpc, normalization, indices, sigma=sigma, device=device)
        self.occupant = deepcopy(occupant)
        self.occupancy_model = occupancy_model
        self.occupancy_series = occupancy_series
        self.alpha_occ = float(alpha_occ)
        self.use_bspline_gate = bool(use_bspline_gate)
        self.sim_start = int(sim_start)
        self.T_dyn_init_mode = T_dyn_init_mode
        self.delta_up = float(delta_up)
        self.delta_down = float(delta_down)
        self.drift_to_pref = float(drift_to_pref)
        self.T_dyn_min = float(T_dyn_min)
        self.T_dyn_max = float(T_dyn_max)
        self.T_dyn = None
        self.last_occ_now = 0.0
        self.last_occ_hat = 0.0
        self.last_occ_bin = 0.0

    def reset(self) -> None:
        super().reset()
        self.T_dyn = None
        self._step_count = 0
        self.last_occ_now = 0.0
        self.last_occ_hat = 0.0
        self.last_occ_bin = 0.0

    def snapshot_state(self) -> dict[str, Any]:
        state = super().snapshot_state()
        state["T_dyn"] = None if self.T_dyn is None else float(self.T_dyn)
        state["step_count"] = int(getattr(self, "_step_count", 0))
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        self.T_dyn = state.get("T_dyn", None)
        self._step_count = int(state.get("step_count", getattr(self, "_step_count", 0)))

    def initialize_tdyn_from_env(self, env):
        import numpy as np

        b = env.unwrapped.buildings[0]
        scheduled_now = float(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[b.time_step]
        )
        if self.T_dyn_init_mode == "schedule":
            self.T_dyn = scheduled_now
        elif self.T_dyn_init_mode == "pref":
            self.T_dyn = float(self.occupant.T_pref)
        else:
            raise ValueError("T_dyn_init_mode must be 'schedule' or 'pref'.")
        self.T_dyn = float(np.clip(self.T_dyn, self.T_dyn_min, self.T_dyn_max))

    def update_tdyn_from_feedback(self, feedback: int):
        import numpy as np

        if self.T_dyn is None:
            return
        if self.occupant is not None and self.drift_to_pref > 0.0:
            self.T_dyn = (1.0 - self.drift_to_pref) * self.T_dyn + self.drift_to_pref * float(self.occupant.T_pref)
        if int(feedback) > 0:
            self.T_dyn += self.delta_up
        elif int(feedback) < 0:
            self.T_dyn -= self.delta_down
        self.T_dyn = float(np.clip(self.T_dyn, self.T_dyn_min, self.T_dyn_max))

    def _occupancy_gate_from_step(self, step_count: int):
        import numpy as np
        import torch

        if not self.use_bspline_gate or self.occupancy_model is None or self.occupancy_series is None:
            return torch.ones((1, self.mpc.T, 1), dtype=torch.float32, device=self.device)

        absolute_hint = self.sim_start + int(step_count)
        current_occupancy = int(self.occupancy_series[min(absolute_hint, len(self.occupancy_series) - 1)])
        occ_hat_h, occ_bin_h = forecast_occupancy_from_bsplines(
            self.occupancy_model,
            current_occupancy=current_occupancy,
            time_step=absolute_hint,
            horizon=max(1, self.mpc.T - 1),
            alpha=self.alpha_occ,
        )
        gate = np.zeros(self.mpc.T, dtype=np.float32)
        gate[0] = float(current_occupancy > self.alpha_occ)
        if self.mpc.T > 1:
            gate[1:] = occ_bin_h[: self.mpc.T - 1]
        self.last_occ_now = float(current_occupancy)
        self.last_occ_hat = float(occ_hat_h[0]) if len(occ_hat_h) else 0.0
        self.last_occ_bin = float(occ_bin_h[0]) if len(occ_bin_h) else 0.0
        return torch.tensor(gate, dtype=torch.float32, device=self.device).view(1, self.mpc.T, 1)

    def _mpc_mean_action(self, obs, policy_state=None, update_internal: bool = True):
        import numpy as np
        import torch

        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.indices
        Tin = float(o[idx["tin"]])
        Tout = float(o[idx["tout"]])
        Isol = float(o[idx["solar"]])
        state_in = self.snapshot_state() if policy_state is None else policy_state
        state_work = self._updated_offset_state(Tin, Tout, Isol, state_in)
        state_work["T_dyn"] = state_in.get("T_dyn", self.T_dyn)
        state_work["step_count"] = int(state_in.get("step_count", getattr(self, "_step_count", 0)))
        T_dyn = state_work.get("T_dyn", None)
        if T_dyn is None:
            raise RuntimeError("T_dyn not initialized. Call initialize_tdyn_from_env(env) after env.reset().")

        x_t = torch.tensor(
            [[(Tin - self.normalization["x_mean"]) / self.normalization["x_std"]]],
            dtype=torch.float32,
            device=self.device,
        )
        d_phys = d_forecast_from_obs(o, self.mpc.T, indices=idx, device=self.device)
        d_seq = norm_d(d_phys, self.normalization).view(1, self.mpc.T, 2)
        sp_now = (float(T_dyn) - self.normalization["x_mean"]) / self.normalization["x_std"]
        sp_seq = torch.full((1, self.mpc.T, 1), sp_now, dtype=torch.float32, device=self.device)
        w_seq = torch.full((1, self.mpc.T, 1), float(state_work["w_est"]), dtype=torch.float32, device=self.device)
        q_mult_seq = self._occupancy_gate_from_step(int(state_work["step_count"]))
        u0, _, _ = self.mpc(x_t, d_seq, sp_seq, w_seq, q_mult_seq)
        mu = torch.clamp(u0.squeeze(), 0.0, 1.0)
        next_state = {
            "w_est": float(state_work["w_est"]),
            "prev_T": Tin,
            "prev_u": float(mu.item()),
            "prev_Tout": Tout,
            "prev_Isol": Isol,
            "T_dyn": float(T_dyn),
            "step_count": int(state_work["step_count"]) + 1,
        }
        if update_internal:
            self.restore_state(next_state)
        return mu, next_state


class DiffMPCPolicyRawTDyn(DiffMPCPolicyRaw):
    """Deterministic raw policy with persistent T_dyn and optional occupancy gating."""

    def __init__(
        self,
        mpc,
        normalization: dict[str, float],
        indices: dict[str, Any],
        *,
        device,
        occupant,
        occupancy_model=None,
        occupancy_series=None,
        alpha_occ: float = 0.5,
        use_bspline_gate: bool = True,
        sim_start: int = FEB_START,
        T_dyn_init_mode: str = "schedule",
        delta_up: float = 0.5,
        delta_down: float = 0.5,
        drift_to_pref: float = 0.0,
        T_dyn_min: float = 18.0,
        T_dyn_max: float = 26.0,
    ):
        super().__init__(mpc, normalization, indices, device=device)
        self.occupant = deepcopy(occupant)
        self.occupancy_model = occupancy_model
        self.occupancy_series = occupancy_series
        self.alpha_occ = float(alpha_occ)
        self.use_bspline_gate = bool(use_bspline_gate)
        self.sim_start = int(sim_start)
        self.T_dyn_init_mode = T_dyn_init_mode
        self.delta_up = float(delta_up)
        self.delta_down = float(delta_down)
        self.drift_to_pref = float(drift_to_pref)
        self.T_dyn_min = float(T_dyn_min)
        self.T_dyn_max = float(T_dyn_max)
        self.T_dyn = None
        self._step_count = 0
        self.last_occ_now = 0.0
        self.last_occ_hat = 0.0
        self.last_occ_bin = 0.0

    def reset(self):
        super().reset()
        self.T_dyn = None
        self._step_count = 0
        self.last_occ_now = 0.0
        self.last_occ_hat = 0.0
        self.last_occ_bin = 0.0

    def initialize_tdyn_from_env(self, env):
        import numpy as np

        b = env.unwrapped.buildings[0]
        scheduled_now = float(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[b.time_step]
        )
        if self.T_dyn_init_mode == "schedule":
            self.T_dyn = scheduled_now
        elif self.T_dyn_init_mode == "pref":
            self.T_dyn = float(self.occupant.T_pref)
        else:
            raise ValueError("T_dyn_init_mode must be 'schedule' or 'pref'.")
        self.T_dyn = float(np.clip(self.T_dyn, self.T_dyn_min, self.T_dyn_max))

    def update_tdyn_from_feedback(self, feedback: int):
        import numpy as np

        if self.T_dyn is None:
            return
        if self.occupant is not None and self.drift_to_pref > 0.0:
            self.T_dyn = (1.0 - self.drift_to_pref) * self.T_dyn + self.drift_to_pref * float(self.occupant.T_pref)
        if int(feedback) > 0:
            self.T_dyn += self.delta_up
        elif int(feedback) < 0:
            self.T_dyn -= self.delta_down
        self.T_dyn = float(np.clip(self.T_dyn, self.T_dyn_min, self.T_dyn_max))

    def _occupancy_gate(self):
        import numpy as np
        import torch

        if not self.use_bspline_gate or self.occupancy_model is None or self.occupancy_series is None:
            return torch.ones((1, self.mpc.T, 1), dtype=torch.float32, device=self.device)
        time_step = self.sim_start + self._step_count
        current_occupancy = int(self.occupancy_series[min(time_step, len(self.occupancy_series) - 1)])
        occ_hat_h, occ_bin_h = forecast_occupancy_from_bsplines(
            self.occupancy_model,
            current_occupancy=current_occupancy,
            time_step=time_step,
            horizon=max(1, self.mpc.T - 1),
            alpha=self.alpha_occ,
        )
        gate = np.zeros(self.mpc.T, dtype=np.float32)
        gate[0] = float(current_occupancy > self.alpha_occ)
        if self.mpc.T > 1:
            gate[1:] = occ_bin_h[: self.mpc.T - 1]
        self.last_occ_now = float(current_occupancy)
        self.last_occ_hat = float(occ_hat_h[0]) if len(occ_hat_h) else 0.0
        self.last_occ_bin = float(occ_bin_h[0]) if len(occ_bin_h) else 0.0
        return torch.tensor(gate, dtype=torch.float32, device=self.device).view(1, self.mpc.T, 1)

    def predict(self, obs):
        import numpy as np
        import torch

        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.indices
        Tin = float(o[idx["tin"]])
        Tout = float(o[idx["tout"]])
        Isol = float(o[idx["solar"]])
        if self.T_dyn is None:
            raise RuntimeError("T_dyn not initialized. Call initialize_tdyn_from_env(env) after env.reset().")

        x = (Tin - self.normalization["x_mean"]) / self.normalization["x_std"]
        d0_phys = torch.tensor([[Tout, Isol]], dtype=torch.float32, device=self.device)
        d0 = self._norm_d(d0_phys)

        if self.prev_x is not None:
            C = float(self.mpc._pos(self.mpc.C_raw))
            Rm = float(self.mpc._pos(self.mpc.Rm_raw))
            Rout = float(self.mpc._pos(self.mpc.Rout_raw))
            Aeff = float(self.mpc._pos(self.mpc.Aeff_raw))
            Pnom = float(self.mpc._pos(self.mpc.Pnom_raw))
            Ac = -(1 / (Rm * C) + 1 / (Rout * C))
            A = np.exp(Ac * self.mpc.dt)
            phi = (A - 1.0) / Ac if abs(Ac) > 1e-8 else self.mpc.dt
            tout_prev = float(self.prev_d[0, 0])
            isol_prev = float(self.prev_d[0, 1])
            cop = float(self.mpc.cop_a + self.mpc.cop_b * tout_prev)
            cop = np.clip(cop, self.mpc.cop_min, self.mpc.cop_max)
            Bu = phi * (cop * Pnom / C)
            Bd0 = phi * (1 / (Rm * C))
            Bd1 = phi * (1 / (Rout * C))
            Bd2 = phi * (Aeff / C)
            x_pred = (
                A * self.prev_x
                + Bu * self.prev_u
                + Bd0 * float(self.mpc.Tm.item())
                + Bd1 * tout_prev
                + Bd2 * isol_prev
                + self.w
            )
            self.w += self.gamma * (x - x_pred)

        x_t = torch.tensor([[x]], dtype=torch.float32, device=self.device)
        d_phys = d_forecast_from_obs(o, self.mpc.T, indices=idx, device=self.device)
        d_seq = self._norm_d(d_phys).view(1, self.mpc.T, 2)
        sp_now = (float(self.T_dyn) - self.normalization["x_mean"]) / self.normalization["x_std"]
        sp_seq = torch.full((1, self.mpc.T, 1), sp_now, dtype=torch.float32, device=self.device)
        w_seq = torch.full((1, self.mpc.T, 1), float(self.w), dtype=torch.float32, device=self.device)
        q_mult_seq = self._occupancy_gate()

        with torch.no_grad():
            u0, _, _ = self.mpc(x_t, d_seq, sp_seq, w_seq, q_mult_seq)

        u = float(u0.item())
        self.prev_x = x
        self.prev_u = u
        self.prev_d = d0
        self.last_history_bias = 0.0
        self._step_count += 1
        return [np.array([u], dtype=np.float32)]


def collect_rollout(env, policy, *, seed: int = 49, max_steps: int | None = None) -> list[dict[str, Any]]:
    import numpy as np

    set_experiment_seed(seed)
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "initialize_tdyn_from_env"):
        policy.initialize_tdyn_from_env(env)

    trajectory = []
    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated):
        action_env, logp, a_sample, mu, policy_state = policy.act(obs)
        baseline_state = {
            "decision_w_est": float(getattr(policy, "w_est", 0.0)),
            "prev_u": None if getattr(policy, "prev_u", None) is None else float(getattr(policy, "prev_u")),
            "history_bias": float(getattr(policy, "last_history_bias", 0.0)),
        }
        if getattr(policy, "T_dyn", None) is not None:
            baseline_state["T_dyn"] = float(getattr(policy, "T_dyn"))
        next_obs, reward, terminated, truncated, _ = env.step(action_env)
        feedback = int(getattr(env, "last_feedback", 0))
        if hasattr(policy, "update_tdyn_from_feedback"):
            policy.update_tdyn_from_feedback(feedback)
        reward_value = float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward)
        trajectory.append(
            {
                "obs": obs,
                "policy_state": policy_state,
                "baseline_state": baseline_state,
                "a_sample": a_sample,
                "logp_old": logp,
                "mu_old": mu,
                "reward": reward_value,
                "feedback": feedback,
            }
        )
        obs = next_obs
        step_count += 1
        if max_steps is not None and step_count >= int(max_steps):
            break
    return trajectory


def ppo_update_mpc(
    mpc,
    trajectory: list[dict[str, Any]],
    advantages,
    *,
    normalization: dict[str, float],
    indices: dict[str, Any],
    device,
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr: float = 5e-3,
    max_grad_norm: float = 1.0,
    policy_cls=None,
    policy_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    import numpy as np
    import torch

    from src.controllers.rlmpc import freeze_dynamics_parameters

    mpc.train()
    freeze_dynamics_parameters(mpc)
    opt = torch.optim.Adam([mpc.q_track_raw, mpc.r_u_raw, mpc.sp_bias_raw], lr=float(lr))

    n_steps = len(trajectory)
    idxs = np.arange(n_steps)
    logp_old = torch.tensor([step["logp_old"] for step in trajectory], dtype=torch.float32, device=device)
    a_sample = torch.tensor([step["a_sample"] for step in trajectory], dtype=torch.float32, device=device)
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    policy_cls = StochasticDiffMPCPolicy if policy_cls is None else policy_cls
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    policy = policy_cls(mpc, normalization, indices, sigma=sigma, device=device, **policy_kwargs)
    history = []

    for ep in range(int(ppo_epochs)):
        np.random.shuffle(idxs)
        ep_losses = []
        for start in range(0, n_steps, int(minibatch_size)):
            mb = idxs[start : start + int(minibatch_size)]
            loss = 0.0
            for j in mb:
                logp_new_j, _mu = policy.log_prob(
                    trajectory[j]["obs"],
                    a_sample[j],
                    policy_state=trajectory[j]["policy_state"],
                )
                ratio = torch.exp(logp_new_j - logp_old[j])
                unclipped = ratio * adv[j]
                clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv[j]
                loss = loss + (-torch.min(unclipped, clipped))
            loss = loss / max(1, len(mb))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mpc.q_track_raw, mpc.r_u_raw, mpc.sp_bias_raw], max_grad_norm)
            opt.step()
            ep_losses.append(float(loss.item()))

        row = {
            "ppo_epoch": ep + 1,
            "loss": float(np.mean(ep_losses)),
            "q_track": float(mpc.q_track().detach().cpu()),
            "r_u": float(mpc.r_u().detach().cpu()),
            "sp_bias": float(mpc.sp_bias().detach().cpu()),
        }
        history.append(row)
        print(
            f"PPO epoch {ep + 1:02d}/{int(ppo_epochs)} | loss {row['loss']:.6f} | "
            f"q_track={row['q_track']:.4f} r_u={row['r_u']:.4f} sp_bias={row['sp_bias']:.4f}"
        )

    mpc.eval()
    return history


def make_raw_env_window(*, sim_start: int, sim_end: int, seed: int = 49, reward_function=None):
    from src.envs import EnvConfig, make_citylearn_env

    dataset_dir = Path("data") / "datasets" / DATASET_NAME
    config = EnvConfig(
        schema_path=dataset_dir / "schema.json",
        dataset_dir=dataset_dir,
        buildings=[CONTROL_BUILDING_NAME],
        start_time_step=int(sim_start),
        end_time_step=int(sim_end),
        random_seed=int(seed),
        central_agent=True,
        active_actions=("heating_device",),
    )
    overrides = {}
    if reward_function is not None:
        overrides["reward_function"] = reward_function
    return make_citylearn_env(config, **overrides)


def make_february_raw_env(*, seed: int = 49, reward_function=None):
    return make_raw_env_window(sim_start=FEB_START, sim_end=FEB_END, seed=seed, reward_function=reward_function)


def _empty_kpis():
    import pandas as pd

    return pd.DataFrame(columns=["cost_function", "value", "level", "name"])


def _get_citylearn_kpi(district_kpis, name: str):
    rows = district_kpis[district_kpis["cost_function"] == name]["value"]
    return float(rows.iloc[0]) if len(rows) > 0 else None


def evaluate_diffmpc_policy(
    *,
    mpc,
    normalization: dict[str, float],
    seed: int,
    reward_function=None,
    device,
    max_steps: int | None = None,
    occupant=None,
    policy_cls=None,
    policy_kwargs: dict[str, Any] | None = None,
    label: str = "RLMPC",
) -> dict[str, Any]:
    import numpy as np

    env = make_february_raw_env(seed=seed, reward_function=reward_function)
    if occupant is not None:
        env = OccupantWrapperWithFeedback(env, deepcopy(occupant))
    obs, _ = env.reset(seed=seed)
    building = env.unwrapped.buildings[0]
    indices = observation_indices_from_names(list(building.active_observations))
    policy_cls = DiffMPCPolicyRaw if policy_cls is None else policy_cls
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    policy = policy_cls(mpc, normalization, indices, device=device, **policy_kwargs)
    policy.reset()
    if hasattr(policy, "initialize_tdyn_from_env"):
        policy.initialize_tdyn_from_env(env)

    terminated = False
    truncated = False
    step_count = 0
    time_hist = []
    Tin_hist = []
    Tmin_hist = []
    Tmax_hist = []
    u_hist = []
    price_hist = []
    reward_hist = []
    baseline_setpoints = []
    effective_setpoints = []
    net_electricity = []
    feedback_hist = []
    tdyn_hist = []
    occ_now_hist = []
    occ_hat_hist = []
    occ_bin_hist = []

    while not (terminated or truncated):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        time_step = int(building.time_step)
        Tin = float(o[indices["tin"]])
        Tmin = float(o[indices["tsp"]])
        Tmax = float(o[indices["tmax"]]) if indices.get("tmax") is not None else 24.0
        price = float(building.pricing.electricity_pricing[time_step])
        scheduled = float(
            building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[time_step]
        )
        effective = float(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[time_step])
        action = policy.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        feedback = int(getattr(env, "last_feedback", 0))
        if hasattr(policy, "update_tdyn_from_feedback"):
            policy.update_tdyn_from_feedback(feedback)

        time_hist.append(step_count)
        Tin_hist.append(Tin)
        Tmin_hist.append(Tmin)
        Tmax_hist.append(Tmax)
        u_hist.append(float(action[0][0]))
        price_hist.append(price)
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
        baseline_setpoints.append(scheduled)
        effective_setpoints.append(effective)
        feedback_hist.append(float(feedback))
        if getattr(policy, "T_dyn", None) is not None:
            tdyn_hist.append(float(getattr(policy, "T_dyn")))
        if hasattr(policy, "last_occ_now"):
            occ_now_hist.append(float(policy.last_occ_now))
            occ_hat_hist.append(float(policy.last_occ_hat))
            occ_bin_hist.append(float(policy.last_occ_bin))
        if hasattr(building, "net_electricity_consumption"):
            net_electricity.append(float(building.net_electricity_consumption[-1]))
        step_count += 1
        if max_steps is not None and step_count >= int(max_steps):
            break

    if max_steps is None:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    else:
        kpis = _empty_kpis()
        district_kpis = kpis.copy()

    return {
        "name": label,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "time_step": np.asarray(time_hist, dtype=int),
        "Tin_hist": np.asarray(Tin_hist, dtype=float),
        "Tmin_hist": np.asarray(Tmin_hist, dtype=float),
        "Tmax_hist": np.asarray(Tmax_hist, dtype=float),
        "u_hist": np.asarray(u_hist, dtype=float),
        "price_hist": np.asarray(price_hist, dtype=float),
        "reward_hist": np.asarray(reward_hist, dtype=float),
        "baseline_setpoints": np.asarray(baseline_setpoints, dtype=float),
        "effective_setpoints": np.asarray(effective_setpoints, dtype=float),
        "net_electricity_consumption": np.asarray(net_electricity, dtype=float),
        "feedback_hist": np.asarray(feedback_hist, dtype=float),
        "override_count": int(getattr(env, "override_count", 0)),
        **({"T_dyn_hist": np.asarray(tdyn_hist, dtype=float)} if len(tdyn_hist) else {}),
        **({"occ_now_hist": np.asarray(occ_now_hist, dtype=float)} if len(occ_now_hist) else {}),
        **({"occ_hat_hist": np.asarray(occ_hat_hist, dtype=float)} if len(occ_hat_hist) else {}),
        **({"occ_bin_hist": np.asarray(occ_bin_hist, dtype=float)} if len(occ_bin_hist) else {}),
        **({"target_temperature": float(occupant.T_pref)} if occupant is not None and hasattr(occupant, "T_pref") else {}),
        **(
            {"predicted_occupied_fraction": float(np.mean(occ_bin_hist))}
            if len(occ_bin_hist)
            else {}
        ),
    }


def rlmpc_rollout_history_dataframe(result: dict[str, Any]):
    import numpy as np
    import pandas as pd

    n_steps = len(result["Tin_hist"])
    data = {
        "time_step": result["time_step"],
        "indoor_temperature": result["Tin_hist"],
        "tmin": result["Tmin_hist"],
        "tmax": result["Tmax_hist"],
        "action": result["u_hist"],
        "price": result["price_hist"],
        "reward": result["reward_hist"],
        "baseline_setpoint": result["baseline_setpoints"],
        "effective_setpoint": result["effective_setpoints"],
    }
    if len(result.get("net_electricity_consumption", [])) == n_steps:
        data["net_electricity_consumption"] = np.asarray(result["net_electricity_consumption"], dtype=float)
    optional = {
        "tdyn": "T_dyn_hist",
        "occupant_feedback": "feedback_hist",
        "occ_now": "occ_now_hist",
        "occ_hat_1step": "occ_hat_hist",
        "occ_bin_1step": "occ_bin_hist",
    }
    for column, key in optional.items():
        if key in result and len(result[key]) == n_steps:
            data[column] = np.asarray(result[key], dtype=float)
    return pd.DataFrame(data)


def summarize_rlmpc_run(label: str, result: dict[str, Any], *, online_info: dict[str, Any] | None = None):
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "reward_total": float(result["reward_hist"].sum()) if len(result["reward_hist"]) else 0.0,
        "electricity_consumption_total": _get_citylearn_kpi(result["district_kpis"], "electricity_consumption_total"),
        "discomfort_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_proportion"),
        "discomfort_cold_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_cold_proportion"),
        "discomfort_hot_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_hot_proportion"),
        "daily_peak_average": _get_citylearn_kpi(result["district_kpis"], "daily_peak_average"),
    }
    if online_info:
        summary.update(online_info)
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "predicted_occupied_fraction" in result:
        summary["predicted_occupied_fraction"] = float(result["predicted_occupied_fraction"])
    if "T_dyn_hist" in result and len(result["T_dyn_hist"]):
        summary["tdyn_mean"] = float(result["T_dyn_hist"].mean())
        summary["tdyn_final"] = float(result["T_dyn_hist"][-1])
    return summary


def save_rlmpc_result(
    result: dict[str, Any],
    *,
    output_dir: Path,
    summary_dir: Path,
    scenario: str,
    label: str,
    metrics: dict[str, Any],
):
    import json

    output_case_dir = output_dir / scenario / label
    summary_case_dir = summary_dir / scenario
    output_case_dir.mkdir(parents=True, exist_ok=True)
    summary_case_dir.mkdir(parents=True, exist_ok=True)

    rlmpc_rollout_history_dataframe(result).to_csv(output_case_dir / "rollout.csv", index=False)
    result["kpis"].to_csv(output_case_dir / "kpis.csv", index=False)
    result["district_kpis"].to_csv(output_case_dir / "district_kpis.csv", index=False)
    (summary_case_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def run_online_rlmpc_no_occupant(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path = "results/raw/rlmpc",
    summary_dir: str | Path = "results/summaries/rlmpc",
    seed: int = 49,
    training_seed: int = 0,
    reward_function=None,
    device: str = "auto",
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr_ppo: float = 5e-3,
    gamma: float = 0.99,
    max_steps: int | None = None,
    label: str = "baseline",
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from src.controllers.rlmpc import diffmpc_param_snapshot

    set_experiment_seed(training_seed)
    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    env_probe = make_february_raw_env(seed=seed, reward_function=reward_function)
    obs, _ = env_probe.reset(seed=seed)
    env_indices = observation_indices_from_names(list(env_probe.unwrapped.buildings[0].active_observations))
    base_dim = len(env_probe.unwrapped.buildings[0].active_observations)

    baseline_model = fit_january_value_baseline(
        dataset_path=dataset_path,
        indices=env_indices,
        base_dim=base_dim,
        gamma=gamma,
    )
    print("Baseline V_pi0 trained.")

    env_train = make_february_raw_env(seed=seed, reward_function=reward_function)
    policy_stoch = StochasticDiffMPCPolicy(
        mpc,
        checkpoint["normalization"],
        env_indices,
        sigma=sigma,
        device=torch_device,
    )
    trajectory = collect_rollout(env_train, policy_stoch, seed=seed, max_steps=max_steps)
    advantages, returns = compute_advantages_with_baseline(
        trajectory,
        baseline_model,
        indices=env_indices,
        base_dim=base_dim,
        gamma=gamma,
    )
    print("Collected steps:", len(trajectory))
    print(f"Adv mean: {advantages.mean():.6f}")
    print(f"Adv std : {advantages.std():.6f}")
    print(f"Adv min/max: {advantages.min():.6f} {advantages.max():.6f}")

    params_before = diffmpc_param_snapshot(mpc)
    ppo_history = ppo_update_mpc(
        mpc,
        trajectory,
        advantages,
        normalization=checkpoint["normalization"],
        indices=env_indices,
        device=torch_device,
        sigma=sigma,
        clip_eps=clip_eps,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        lr=lr_ppo,
        max_grad_norm=1.0,
    )
    params_after = diffmpc_param_snapshot(mpc)

    result = evaluate_diffmpc_policy(
        mpc=mpc,
        normalization=checkpoint["normalization"],
        seed=seed,
        reward_function=reward_function,
        device=torch_device,
        max_steps=max_steps,
    )

    online_info = {
        "train_reward_total": float(np.sum([step["reward"] for step in trajectory])),
        "train_steps": int(len(trajectory)),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "q_track_before": float(params_before["q_track"]),
        "r_u_before": float(params_before["r_u"]),
        "sp_bias_before": float(params_before["sp_bias"]),
        "q_track_after": float(params_after["q_track"]),
        "r_u_after": float(params_after["r_u"]),
        "sp_bias_after": float(params_after["sp_bias"]),
    }
    metrics = summarize_rlmpc_run(label, result, online_info=online_info)
    save_rlmpc_result(
        result,
        output_dir=Path(output_dir),
        summary_dir=Path(summary_dir),
        scenario="no_occupant",
        label=label,
        metrics=metrics,
    )

    output_case_dir = Path(output_dir) / "no_occupant" / label
    pd.DataFrame(ppo_history).to_csv(output_case_dir / "ppo_history.csv", index=False)
    updated_checkpoint = {
        **checkpoint,
        "state_dict": {key: value.detach().cpu().clone() for key, value in mpc.state_dict().items()},
        "online_info": online_info,
        "ppo_history": ppo_history,
    }
    online_checkpoint_path = output_case_dir / "online_updated_checkpoint.pt"
    torch.save(updated_checkpoint, online_checkpoint_path)

    summary_path = Path(summary_dir) / "no_occupant" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    print("February reward after PPO:", metrics["reward_total"])
    print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    print(f"Saved rollout to {output_case_dir / 'rollout.csv'}")
    print(f"Saved summary to {summary_path}")
    return {"result": result, "metrics": metrics, "ppo_history": ppo_history}


def _run_online_rlmpc_occupant_case(
    *,
    label: str,
    occupant: Occupant,
    mode: str,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: Path,
    summary_dir: Path,
    seed: int,
    training_seed: int,
    reward_function,
    device: str,
    sigma: float,
    clip_eps: float,
    ppo_epochs: int,
    minibatch_size: int,
    lr_ppo: float,
    gamma: float,
    max_steps: int | None,
    alpha_occ: float,
    T_dyn_init_mode: str,
    delta_up: float,
    delta_down: float,
    drift_to_pref: float,
    T_dyn_min: float,
    T_dyn_max: float,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from src.controllers.rlmpc import diffmpc_param_snapshot

    set_experiment_seed(training_seed)
    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    env_probe = make_february_raw_env(seed=seed, reward_function=reward_function)
    env_probe.reset(seed=seed)
    env_indices = observation_indices_from_names(list(env_probe.unwrapped.buildings[0].active_observations))
    base_dim = len(env_probe.unwrapped.buildings[0].active_observations)

    baseline_model = fit_january_value_baseline(
        dataset_path=dataset_path,
        indices=env_indices,
        base_dim=base_dim,
        gamma=gamma,
    )
    print("Baseline V_pi0 trained.")

    occupancy_model = None
    occupancy_series = None
    if mode == "bspline_tdyn":
        occupancy_model, occupancy_series = fit_bspline_occupancy_model(seed=seed)

    params_before = diffmpc_param_snapshot(mpc)
    param_history = [{"stage": "offline_init", "week": 0, **params_before}]
    weekly_training = []
    ppo_rows = []
    n_weeks = int(np.ceil((FEB_END - FEB_START + 1) / WEEK_HOURS))

    for week_idx in range(n_weeks):
        week_start = FEB_START + week_idx * WEEK_HOURS
        week_end = min(week_start + WEEK_HOURS - 1, FEB_END)
        week_seed = int(seed + week_idx)
        env_train = make_raw_env_window(
            sim_start=week_start,
            sim_end=week_end,
            seed=week_seed,
            reward_function=reward_function,
        )
        env_train = OccupantWrapperWithFeedback(env_train, deepcopy(occupant))

        if mode == "bspline_tdyn":
            policy_stoch = StochasticDiffMPCPolicyTDyn(
                mpc,
                checkpoint["normalization"],
                env_indices,
                sigma=sigma,
                device=torch_device,
                occupant=deepcopy(occupant),
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                alpha_occ=alpha_occ,
                use_bspline_gate=True,
                sim_start=week_start,
                T_dyn_init_mode=T_dyn_init_mode,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                T_dyn_min=T_dyn_min,
                T_dyn_max=T_dyn_max,
            )
            ppo_policy_cls = StochasticDiffMPCPolicyTDyn
            ppo_policy_kwargs = {
                "occupant": deepcopy(occupant),
                "occupancy_model": occupancy_model,
                "occupancy_series": occupancy_series,
                "alpha_occ": alpha_occ,
                "use_bspline_gate": True,
                "sim_start": week_start,
                "T_dyn_init_mode": T_dyn_init_mode,
                "delta_up": delta_up,
                "delta_down": delta_down,
                "drift_to_pref": drift_to_pref,
                "T_dyn_min": T_dyn_min,
                "T_dyn_max": T_dyn_max,
            }
        else:
            policy_stoch = StochasticDiffMPCPolicy(
                mpc,
                checkpoint["normalization"],
                env_indices,
                sigma=sigma,
                device=torch_device,
            )
            ppo_policy_cls = StochasticDiffMPCPolicy
            ppo_policy_kwargs = {}

        trajectory = collect_rollout(env_train, policy_stoch, seed=week_seed, max_steps=max_steps)
        advantages, _returns = compute_advantages_with_baseline(
            trajectory,
            baseline_model,
            indices=env_indices,
            base_dim=base_dim,
            gamma=gamma,
        )
        week_reward = float(np.sum([step["reward"] for step in trajectory]))
        week_overrides = int(getattr(env_train, "override_count", 0))
        print(
            f"{label} | week {week_idx + 1}/{n_weeks} {week_start}-{week_end} | "
            f"reward={week_reward:.3f} | overrides={week_overrides}"
        )

        ppo_history = ppo_update_mpc(
            mpc,
            trajectory,
            advantages,
            normalization=checkpoint["normalization"],
            indices=env_indices,
            device=torch_device,
            sigma=sigma,
            clip_eps=clip_eps,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            lr=lr_ppo,
            max_grad_norm=1.0,
            policy_cls=ppo_policy_cls,
            policy_kwargs=ppo_policy_kwargs,
        )
        for row in ppo_history:
            ppo_rows.append({"week": week_idx + 1, **row})

        params_week = diffmpc_param_snapshot(mpc)
        weekly_training.append(
            {
                "week": week_idx + 1,
                "sim_start": week_start,
                "sim_end": week_end,
                "train_steps": int(len(trajectory)),
                "train_reward": week_reward,
                "train_override_count": week_overrides,
                "advantage_mean": float(advantages.mean()),
                "advantage_std": float(advantages.std()),
            }
        )
        param_history.append(
            {
                "stage": f"week_{week_idx + 1}",
                "week": week_idx + 1,
                "sim_start": week_start,
                "sim_end": week_end,
                **params_week,
            }
        )
        if max_steps is not None:
            break

    params_after = diffmpc_param_snapshot(mpc)
    eval_seed = int(seed + 10_000)
    if mode == "bspline_tdyn":
        eval_policy_cls = DiffMPCPolicyRawTDyn
        eval_policy_kwargs = {
            "occupant": deepcopy(occupant),
            "occupancy_model": occupancy_model,
            "occupancy_series": occupancy_series,
            "alpha_occ": alpha_occ,
            "use_bspline_gate": True,
            "sim_start": FEB_START,
            "T_dyn_init_mode": T_dyn_init_mode,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "drift_to_pref": drift_to_pref,
            "T_dyn_min": T_dyn_min,
            "T_dyn_max": T_dyn_max,
        }
    else:
        eval_policy_cls = DiffMPCPolicyRaw
        eval_policy_kwargs = {}

    result = evaluate_diffmpc_policy(
        mpc=mpc,
        normalization=checkpoint["normalization"],
        seed=eval_seed,
        reward_function=reward_function,
        device=torch_device,
        max_steps=max_steps,
        occupant=deepcopy(occupant),
        policy_cls=eval_policy_cls,
        policy_kwargs=eval_policy_kwargs,
        label=label,
    )

    online_info = {
        "train_reward_total": float(np.sum([row["train_reward"] for row in weekly_training])),
        "train_steps": int(np.sum([row["train_steps"] for row in weekly_training])),
        "q_track_before": float(params_before["q_track"]),
        "r_u_before": float(params_before["r_u"]),
        "sp_bias_before": float(params_before["sp_bias"]),
        "q_track_after": float(params_after["q_track"]),
        "r_u_after": float(params_after["r_u"]),
        "sp_bias_after": float(params_after["sp_bias"]),
    }
    metrics = summarize_rlmpc_run(label, result, online_info=online_info)
    save_rlmpc_result(
        result,
        output_dir=output_dir,
        summary_dir=summary_dir,
        scenario="occupant_present",
        label=label,
        metrics=metrics,
    )

    output_case_dir = output_dir / "occupant_present" / label
    pd.DataFrame(ppo_rows).to_csv(output_case_dir / "ppo_history.csv", index=False)
    pd.DataFrame(weekly_training).to_csv(output_case_dir / "weekly_training.csv", index=False)
    pd.DataFrame(param_history).to_csv(output_case_dir / "param_history.csv", index=False)
    updated_checkpoint = {
        **checkpoint,
        "state_dict": {key: value.detach().cpu().clone() for key, value in mpc.state_dict().items()},
        "online_info": online_info,
        "ppo_history": ppo_rows,
        "weekly_training": weekly_training,
        "param_history": param_history,
    }
    torch.save(updated_checkpoint, output_case_dir / "online_updated_checkpoint.pt")

    print("Reward:", metrics["reward_total"])
    print("Overrides:", metrics["override_count"])
    if mode == "bspline_tdyn" and "tdyn_final" in metrics:
        print(f"T_dyn mean/final: {metrics['tdyn_mean']:.3f} / {metrics['tdyn_final']:.3f}")
    print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return {"result": result, "metrics": metrics, "ppo_history": ppo_rows}


def run_online_rlmpc_occupants(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path = "results/raw/rlmpc",
    summary_dir: str | Path = "results/summaries/rlmpc",
    seed: int = 49,
    training_seed: int = 0,
    reward_function=None,
    device: str = "auto",
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr_ppo: float = 5e-3,
    gamma: float = 0.99,
    max_steps: int | None = None,
    occupant: str = "all",
    occupant_mode: str = "both",
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.0,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
) -> dict[str, Any]:
    import pandas as pd

    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir)
    occupants = fitted_occupants()
    if occupant != "all":
        occupants = {occupant: occupants[occupant]}

    mode_names = []
    if occupant_mode in ("without_tdyn", "both"):
        mode_names.append("without_tdyn")
    if occupant_mode in ("bspline_tdyn", "both"):
        mode_names.append("bspline_tdyn")

    all_results: dict[str, Any] = {}
    summary_rows = []
    for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
        for mode in mode_names:
            label = f"{name}_{mode}"
            print(f"\n{label}")
            case = _run_online_rlmpc_occupant_case(
                label=label,
                occupant=deepcopy(occ),
                mode=mode,
                checkpoint_path=checkpoint_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                summary_dir=summary_dir,
                seed=int(seed + occ_idx),
                training_seed=int(training_seed + occ_idx),
                reward_function=reward_function,
                device=device,
                sigma=sigma,
                clip_eps=clip_eps,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                lr_ppo=lr_ppo,
                gamma=gamma,
                max_steps=max_steps,
                alpha_occ=alpha_occ,
                T_dyn_init_mode=T_dyn_init_mode,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                T_dyn_min=T_dyn_min,
                T_dyn_max=T_dyn_max,
            )
            all_results[label] = case
            summary_rows.append(case["metrics"])

    summary_path = summary_dir / "occupant_present" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"Saved summary to {summary_path}")
    return all_results
