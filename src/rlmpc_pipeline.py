"""High-level RLMPC/Gnu-RL pipeline contracts and offline imitation helpers."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.occupants.metrics import (
    kelvin_hours,
    occupant_probability_comfort_band,
    override_hours,
    setpoint_kelvin_hours,
)
from src.occupants.setpoint_schedule import (
    active_override_steps,
    apply_heating_override_schedule,
    reset_future_heating_setpoints,
)
from src.occupants.tdyn import TDynConfig, update_tdyn_reference


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
                reset_future_heating_setpoints(b, b.time_step)
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

                apply_heating_override_schedule(
                    b,
                    next_ix,
                    new_Tsp,
                    active_override_steps(self.occupant),
                )
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

    occ_prob_h, _ = occupancy_model.predict_occupied_probability(
        current_state=int(current_occupancy),
        start_t=int(time_step),
        horizon=int(horizon),
    )
    occ_prob_h = np.clip(np.asarray(occ_prob_h, dtype=float), 0.0, 1.0)
    occ_gate_h = (occ_prob_h > float(alpha)).astype(float)
    return occ_prob_h, occ_gate_h


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


def _schedule_reference_tensor(
    schedule_profile,
    *,
    sim_start: int,
    step_count: int,
    horizon: int,
    fallback_schedule: float,
    device,
):
    import numpy as np
    import torch

    if schedule_profile is None:
        values = np.full(int(horizon), float(fallback_schedule), dtype=np.float32)
    else:
        profile = np.asarray(schedule_profile, dtype=float)
        values = np.asarray(
            [
                profile[min(int(sim_start) + int(step_count) + h, len(profile) - 1)]
                for h in range(int(horizon))
            ],
            dtype=np.float32,
        )
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, int(horizon), 1)


def _tdyn_reference_sequence(
    *,
    schedule_profile,
    effective_setpoint_profile=None,
    sim_start: int,
    step_count: int,
    horizon: int,
    fallback_schedule: float,
    T_dyn: float,
    occupancy_gate,
    normalization: dict[str, float],
    device,
):
    import torch

    schedule = _schedule_reference_tensor(
        schedule_profile,
        sim_start=sim_start,
        step_count=step_count,
        horizon=horizon,
        fallback_schedule=fallback_schedule,
        device=device,
    )
    if occupancy_gate is None:
        return norm_sp(schedule, normalization)
    gate = occupancy_gate.to(device=device, dtype=schedule.dtype)
    tdyn = torch.full_like(schedule, float(T_dyn))
    reference = (1.0 - gate) * schedule + gate * tdyn
    return norm_sp(reference, normalization)


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
    building_name: str | None = None,
    building_metadata: dict[str, Any] | None = None,
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
    if building_name is not None:
        checkpoint["building_name"] = str(building_name)
    if building_metadata:
        checkpoint.update(building_metadata)

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

    def __init__(
        self,
        mpc,
        normalization: dict[str, float],
        indices: dict[str, Any],
        *,
        device,
        stress_profile=None,
        future_stress_profile=None,
        setpoint_bias_profile=None,
        district_target_profile=None,
        district_background_profile=None,
        district_action_power_profile=None,
    ):
        self.mpc = mpc
        self.normalization = normalization
        self.indices = indices
        self.device = device
        self.stress_profile = None if stress_profile is None else [float(v) for v in stress_profile]
        self.future_stress_profile = (
            None if future_stress_profile is None else [float(v) for v in future_stress_profile]
        )
        self.setpoint_bias_profile = (
            None if setpoint_bias_profile is None else [float(v) for v in setpoint_bias_profile]
        )
        self.district_target_profile = None if district_target_profile is None else [float(v) for v in district_target_profile]
        self.district_background_profile = (
            None if district_background_profile is None else [float(v) for v in district_background_profile]
        )
        self.district_action_power_profile = (
            None if district_action_power_profile is None else [float(v) for v in district_action_power_profile]
        )
        self.w = 0.0
        self.gamma = 0.2
        self.last_history_bias = 0.0
        self.prev_x = None
        self.prev_u = None
        self.prev_d = None
        self._step_count = 0

    def reset(self):
        self.w = 0.0
        self.prev_x = None
        self.prev_u = None
        self.prev_d = None
        self.last_history_bias = 0.0
        self._step_count = 0

    def _norm_d(self, value):
        return norm_d(value, self.normalization)

    def _stress_seq(self):
        import torch

        if not self.stress_profile:
            return None
        horizon_values = [
            self.stress_profile[min(self._step_count + i, len(self.stress_profile) - 1)]
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _future_stress_seq(self):
        import torch

        if not self.future_stress_profile:
            return None
        horizon_values = [
            self.future_stress_profile[min(self._step_count + i, len(self.future_stress_profile) - 1)]
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _setpoint_bias_seq(self):
        import torch

        if not self.setpoint_bias_profile:
            return None
        x_std = max(float(self.normalization["x_std"]), 1e-6)
        horizon_values = [
            self.setpoint_bias_profile[min(self._step_count + i, len(self.setpoint_bias_profile) - 1)] / x_std
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _district_target_seqs(self, start_step: int | None = None):
        import torch

        if (
            not self.district_target_profile
            or not self.district_background_profile
            or not self.district_action_power_profile
        ):
            return None, None, None
        start = self._step_count if start_step is None else int(start_step)

        def horizon_tensor(values):
            horizon_values = [values[min(start + i, len(values) - 1)] for i in range(self.mpc.T)]
            return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

        return (
            horizon_tensor(self.district_target_profile),
            horizon_tensor(self.district_background_profile),
            horizon_tensor(self.district_action_power_profile),
        )

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

        district_target_seq, district_background_seq, district_action_power_seq = self._district_target_seqs()
        with torch.no_grad():
            u0, _, _ = self.mpc(
                x_t,
                d_seq,
                sp_seq,
                w_seq,
                u_prev=0.0 if self.prev_u is None else self.prev_u,
                stress_seq=self._stress_seq(),
                future_stress_seq=self._future_stress_seq(),
                setpoint_bias_seq=self._setpoint_bias_seq(),
                district_target_seq=district_target_seq,
                district_background_seq=district_background_seq,
                district_action_power_seq=district_action_power_seq,
            )

        u = float(u0.item())
        self.prev_x = x
        self.prev_u = u
        self.prev_d = d0
        self.last_history_bias = 0.0
        self._step_count += 1
        return [np.array([u], dtype=np.float32)]


def make_january_raw_env(
    *,
    seed: int = 49,
    reward_function=None,
    reward_function_kwargs=None,
    building_name: str = CONTROL_BUILDING_NAME,
):
    from src.envs import EnvConfig, make_citylearn_env

    dataset_dir = Path("data") / "datasets" / DATASET_NAME
    config = EnvConfig(
        schema_path=dataset_dir / "schema.json",
        dataset_dir=dataset_dir,
        buildings=[building_name],
        start_time_step=JAN_START,
        end_time_step=JAN_END,
        random_seed=int(seed),
        central_agent=True,
        active_actions=("heating_device",),
    )
    overrides = {}
    if reward_function is not None:
        overrides["reward_function"] = reward_function
    if reward_function_kwargs is not None:
        overrides["reward_function_kwargs"] = reward_function_kwargs
    return make_citylearn_env(config, **overrides)


def run_january_raw_rollout(
    *,
    checkpoint_path: str | Path,
    seed: int = 49,
    reward_function=None,
    device: str = "auto",
    max_steps: int | None = None,
    building_name: str = CONTROL_BUILDING_NAME,
) -> dict[str, Any]:
    import numpy as np

    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    env = make_january_raw_env(seed=seed, reward_function=reward_function, building_name=building_name)
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
        "building_name": building_name,
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

    def __init__(
        self,
        mpc,
        normalization: dict[str, float],
        indices: dict[str, Any],
        *,
        sigma: float,
        device,
        stress_profile=None,
        future_stress_profile=None,
        setpoint_bias_profile=None,
        district_target_profile=None,
        district_background_profile=None,
        district_action_power_profile=None,
    ):
        self.mpc = mpc
        self.normalization = normalization
        self.indices = indices
        self.sigma = float(sigma)
        self.device = device
        self.stress_profile = None if stress_profile is None else [float(v) for v in stress_profile]
        self.future_stress_profile = (
            None if future_stress_profile is None else [float(v) for v in future_stress_profile]
        )
        self.setpoint_bias_profile = (
            None if setpoint_bias_profile is None else [float(v) for v in setpoint_bias_profile]
        )
        self.district_target_profile = None if district_target_profile is None else [float(v) for v in district_target_profile]
        self.district_background_profile = (
            None if district_background_profile is None else [float(v) for v in district_background_profile]
        )
        self.district_action_power_profile = (
            None if district_action_power_profile is None else [float(v) for v in district_action_power_profile]
        )
        self.w_est = 0.0
        self.gamma = 0.2
        self.prev_T = None
        self.prev_u = None
        self.prev_Tout = None
        self.prev_Isol = None
        self.last_history_bias = 0.0
        self._step_count = 0

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "w_est": float(self.w_est),
            "prev_T": None if self.prev_T is None else float(self.prev_T),
            "prev_u": None if self.prev_u is None else float(self.prev_u),
            "prev_Tout": None if self.prev_Tout is None else float(self.prev_Tout),
            "prev_Isol": None if self.prev_Isol is None else float(self.prev_Isol),
            "step_count": int(self._step_count),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.w_est = float(state["w_est"])
        self.prev_T = state["prev_T"]
        self.prev_u = state["prev_u"]
        self.prev_Tout = state["prev_Tout"]
        self.prev_Isol = state["prev_Isol"]
        self._step_count = int(state.get("step_count", self._step_count))

    def reset(self) -> None:
        self.restore_state(
            {
                "w_est": 0.0,
                "prev_T": None,
                "prev_u": None,
                "prev_Tout": None,
                "prev_Isol": None,
                "step_count": 0,
            }
        )
        self.last_history_bias = 0.0

    def _stress_seq(self, start_step: int):
        import torch

        if not self.stress_profile:
            return None
        horizon_values = [
            self.stress_profile[min(int(start_step) + i, len(self.stress_profile) - 1)]
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _future_stress_seq(self, start_step: int):
        import torch

        if not self.future_stress_profile:
            return None
        horizon_values = [
            self.future_stress_profile[min(int(start_step) + i, len(self.future_stress_profile) - 1)]
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _setpoint_bias_seq(self, start_step: int):
        import torch

        if not self.setpoint_bias_profile:
            return None
        x_std = max(float(self.normalization["x_std"]), 1e-6)
        horizon_values = [
            self.setpoint_bias_profile[min(int(start_step) + i, len(self.setpoint_bias_profile) - 1)] / x_std
            for i in range(self.mpc.T)
        ]
        return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

    def _district_target_seqs(self, start_step: int):
        import torch

        if (
            not self.district_target_profile
            or not self.district_background_profile
            or not self.district_action_power_profile
        ):
            return None, None, None

        def horizon_tensor(values):
            horizon_values = [
                values[min(int(start_step) + i, len(values) - 1)]
                for i in range(self.mpc.T)
            ]
            return torch.tensor(horizon_values, dtype=torch.float32, device=self.device).view(1, self.mpc.T)

        return (
            horizon_tensor(self.district_target_profile),
            horizon_tensor(self.district_background_profile),
            horizon_tensor(self.district_action_power_profile),
        )

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
        district_target_seq, district_background_seq, district_action_power_seq = self._district_target_seqs(
            int(state_work.get("step_count", 0))
        )
        u0, _, _ = self.mpc(
            x_t,
            d_seq,
            sp_seq,
            w_seq,
            u_prev=0.0 if state_work["prev_u"] is None else state_work["prev_u"],
            stress_seq=self._stress_seq(int(state_work.get("step_count", 0))),
            future_stress_seq=self._future_stress_seq(int(state_work.get("step_count", 0))),
            setpoint_bias_seq=self._setpoint_bias_seq(int(state_work.get("step_count", 0))),
            district_target_seq=district_target_seq,
            district_background_seq=district_background_seq,
            district_action_power_seq=district_action_power_seq,
        )
        mu = torch.clamp(u0.squeeze(), 0.0, 1.0)
        next_state = {
            "w_est": float(state_work["w_est"]),
            "prev_T": Tin,
            "prev_u": float(mu.item()),
            "prev_Tout": Tout,
            "prev_Isol": Isol,
            "step_count": int(state_work.get("step_count", 0)) + 1,
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
        drift_to_pref: float = 0.05,
        T_dyn_min: float = 18.0,
        T_dyn_max: float = 26.0,
        stress_profile=None,
        future_stress_profile=None,
        setpoint_bias_profile=None,
        district_target_profile=None,
        district_background_profile=None,
        district_action_power_profile=None,
    ):
        super().__init__(
            mpc,
            normalization,
            indices,
            sigma=sigma,
            device=device,
            stress_profile=stress_profile,
            future_stress_profile=future_stress_profile,
            setpoint_bias_profile=setpoint_bias_profile,
            district_target_profile=district_target_profile,
            district_background_profile=district_background_profile,
            district_action_power_profile=district_action_power_profile,
        )
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
        self.schedule_profile = None
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
        self.schedule_profile = np.asarray(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control,
            dtype=float,
        )
        self.effective_setpoint_profile = np.asarray(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point,
            dtype=float,
        )
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
        self.T_dyn = update_tdyn_reference(
            self.T_dyn,
            feedback,
            preferred_temperature=(None if self.occupant is None else float(self.occupant.T_pref)),
            config=TDynConfig(
                delta_up=self.delta_up,
                delta_down=self.delta_down,
                drift_to_pref=self.drift_to_pref,
                lower_bound=self.T_dyn_min,
                upper_bound=self.T_dyn_max,
                anchor_feedback_to_pref=True,
            ),
        )

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
        w_seq = torch.full((1, self.mpc.T, 1), float(state_work["w_est"]), dtype=torch.float32, device=self.device)
        q_gate_seq = self._occupancy_gate_from_step(int(state_work["step_count"]))
        sp_seq = _tdyn_reference_sequence(
            schedule_profile=self.schedule_profile,
            effective_setpoint_profile=getattr(self, "effective_setpoint_profile", None),
            sim_start=self.sim_start,
            step_count=int(state_work["step_count"]),
            horizon=self.mpc.T,
            fallback_schedule=float(o[idx["tsp"]]),
            T_dyn=float(T_dyn),
            occupancy_gate=q_gate_seq,
            normalization=self.normalization,
            device=self.device,
        )
        q_mult_seq = torch.ones_like(q_gate_seq)
        district_target_seq, district_background_seq, district_action_power_seq = self._district_target_seqs(
            int(state_work.get("step_count", 0))
        )
        u0, _, _ = self.mpc(
            x_t,
            d_seq,
            sp_seq,
            w_seq,
            q_mult_seq,
            u_prev=0.0 if state_work["prev_u"] is None else state_work["prev_u"],
            stress_seq=self._stress_seq(int(state_work.get("step_count", 0))),
            future_stress_seq=self._future_stress_seq(int(state_work.get("step_count", 0))),
            setpoint_bias_seq=self._setpoint_bias_seq(int(state_work.get("step_count", 0))),
            district_target_seq=district_target_seq,
            district_background_seq=district_background_seq,
            district_action_power_seq=district_action_power_seq,
        )
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
        drift_to_pref: float = 0.05,
        T_dyn_min: float = 18.0,
        T_dyn_max: float = 26.0,
        stress_profile=None,
        future_stress_profile=None,
        setpoint_bias_profile=None,
        district_target_profile=None,
        district_background_profile=None,
        district_action_power_profile=None,
    ):
        super().__init__(
            mpc,
            normalization,
            indices,
            device=device,
            stress_profile=stress_profile,
            future_stress_profile=future_stress_profile,
            setpoint_bias_profile=setpoint_bias_profile,
            district_target_profile=district_target_profile,
            district_background_profile=district_background_profile,
            district_action_power_profile=district_action_power_profile,
        )
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
        self.schedule_profile = None
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
        self.schedule_profile = np.asarray(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control,
            dtype=float,
        )
        self.effective_setpoint_profile = np.asarray(
            b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point,
            dtype=float,
        )
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
        self.T_dyn = update_tdyn_reference(
            self.T_dyn,
            feedback,
            preferred_temperature=(None if self.occupant is None else float(self.occupant.T_pref)),
            config=TDynConfig(
                delta_up=self.delta_up,
                delta_down=self.delta_down,
                drift_to_pref=self.drift_to_pref,
                lower_bound=self.T_dyn_min,
                upper_bound=self.T_dyn_max,
                anchor_feedback_to_pref=True,
            ),
        )

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
        w_seq = torch.full((1, self.mpc.T, 1), float(self.w), dtype=torch.float32, device=self.device)
        q_gate_seq = self._occupancy_gate()
        sp_seq = _tdyn_reference_sequence(
            schedule_profile=self.schedule_profile,
            effective_setpoint_profile=getattr(self, "effective_setpoint_profile", None),
            sim_start=self.sim_start,
            step_count=self._step_count,
            horizon=self.mpc.T,
            fallback_schedule=float(o[idx["tsp"]]),
            T_dyn=float(self.T_dyn),
            occupancy_gate=q_gate_seq,
            normalization=self.normalization,
            device=self.device,
        )
        q_mult_seq = torch.ones_like(q_gate_seq)
        district_target_seq, district_background_seq, district_action_power_seq = self._district_target_seqs(
            self._step_count
        )

        with torch.no_grad():
            u0, _, _ = self.mpc(
                x_t,
                d_seq,
                sp_seq,
                w_seq,
                q_mult_seq,
                u_prev=0.0 if self.prev_u is None else self.prev_u,
                stress_seq=self._stress_seq(),
                future_stress_seq=self._future_stress_seq(),
                setpoint_bias_seq=self._setpoint_bias_seq(),
                district_target_seq=district_target_seq,
                district_background_seq=district_background_seq,
                district_action_power_seq=district_action_power_seq,
            )

        u = float(u0.item())
        self.prev_x = x
        self.prev_u = u
        self.prev_d = d0
        self.last_history_bias = 0.0
        self._step_count += 1
        return [np.array([u], dtype=np.float32)]


def _set_reward_reference_setpoint(env, reference: float | None) -> None:
    reward_function = getattr(env.unwrapped, "reward_function", None)
    if reward_function is None:
        return
    if hasattr(reward_function, "set_current_reference_setpoints"):
        reward_function.set_current_reference_setpoints([None if reference is None else float(reference)])
    if reference is None:
        if hasattr(reward_function, "_t_dyn"):
            reward_function._t_dyn = [None]
    else:
        reward_function._t_dyn = [float(reference)]


def _policy_reward_reference(env, policy) -> float | None:
    import numpy as np

    if getattr(policy, "T_dyn", None) is None:
        return None

    building = env.unwrapped.buildings[0]
    reward_time_step = min(
        int(building.time_step) + 1,
        len(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point) - 1,
    )
    scheduled = float(
        building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
            reward_time_step
        ]
    )
    gate = float(np.clip(getattr(policy, "last_occ_bin", 0.0), 0.0, 1.0))
    return (1.0 - gate) * scheduled + gate * float(policy.T_dyn)


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
            _set_reward_reference_setpoint(env, _policy_reward_reference(env, policy))
        next_obs, reward, terminated, truncated, _ = env.step(action_env)
        feedback = int(getattr(env, "last_feedback", 0))
        if hasattr(policy, "update_tdyn_from_feedback"):
            policy.update_tdyn_from_feedback(feedback)
        reward_value = float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward)
        trajectory.append(
            {
                "obs": obs,
                "next_obs": next_obs,
                "action_env": float(np.asarray(action_env, dtype=np.float32).reshape(-1)[0]),
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


def online_update_diffmpc_dynamics(
    mpc,
    trajectory: list[dict[str, Any]],
    *,
    normalization: dict[str, float],
    indices: dict[str, Any],
    device,
    epochs: int = 5,
    minibatch_size: int = 64,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    validation_fraction: float = 0.2,
    patience: int = 3,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Adapt differentiable 1R1C dynamics from online one-step Tin errors.

    The MPC objective stays fixed; only the RC transition parameters are updated.
    This is a supervised model-adaptation step, not a PPO cost-weight update.
    """

    import numpy as np
    import torch

    from src.controllers.rlmpc import (
        diffmpc_param_snapshot,
        freeze_cost_parameters,
        trainable_dynamics_parameters,
    )

    if not trajectory:
        return []

    mpc.train()
    freeze_cost_parameters(mpc)
    parameters = trainable_dynamics_parameters(mpc)
    opt = torch.optim.Adam(parameters, lr=float(lr))
    x_std = float(normalization["x_std"])

    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(len(trajectory))
    rng.shuffle(all_indices)
    val_size = int(round(len(all_indices) * float(validation_fraction)))
    val_size = min(max(val_size, 1), max(len(all_indices) - 1, 0)) if len(all_indices) >= 8 else 0
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:] if val_size else all_indices

    def _sample_loss(step_idx: int):
        step = trajectory[int(step_idx)]
        obs = np.asarray(step["obs"], dtype=np.float32).reshape(-1)
        next_obs = np.asarray(step["next_obs"], dtype=np.float32).reshape(-1)
        x_t = torch.tensor(
            [[(float(obs[indices["tin"]]) - normalization["x_mean"]) / x_std]],
            dtype=torch.float32,
            device=device,
        )
        x_next = torch.tensor(
            [[(float(next_obs[indices["tin"]]) - normalization["x_mean"]) / x_std]],
            dtype=torch.float32,
            device=device,
        )
        u_t = torch.tensor(
            [[float(step.get("action_env", np.clip(step.get("a_sample", 0.0), 0.0, 1.0)))]],
            dtype=torch.float32,
            device=device,
        )
        d_phys = d_forecast_from_obs(obs, mpc.T, indices=indices, device=device)
        d_seq = norm_d(d_phys, normalization).view(1, mpc.T, 2)
        pred = one_step_pred(mpc, x_t, u_t, d_seq)
        return torch.mean((pred - x_next) ** 2)

    def _evaluate(split_indices):
        if len(split_indices) == 0:
            return {"loss_mean": float("nan"), "loss_max": float("nan"), "rmse_c": float("nan")}
        losses = []
        with torch.no_grad():
            for step_idx in split_indices:
                losses.append(float(_sample_loss(int(step_idx)).detach().cpu()))
        mean_loss = float(np.mean(losses))
        return {
            "loss_mean": mean_loss,
            "loss_max": float(np.max(losses)),
            "rmse_c": float(np.sqrt(max(mean_loss, 0.0)) * x_std),
        }

    def _project_parameters():
        with torch.no_grad():
            for raw_parameter in [mpc.C_raw, mpc.Rm_raw, mpc.Rout_raw, mpc.Aeff_raw, mpc.Pnom_raw]:
                raw_parameter.clamp_(-20.0, 20.0)
            mpc.Tm.clamp_(-10.0, 10.0)
            mpc.cop_a.clamp_(float(mpc.cop_min), float(mpc.cop_max))
            mpc.cop_b.clamp_(-2.0, 2.0)

    history = []
    best_state = _state_dict_cpu(mpc)
    best_score = _evaluate(val_indices if len(val_indices) else train_indices)["loss_mean"]
    stale_epochs = 0

    initial_train = _evaluate(train_indices)
    initial_val = _evaluate(val_indices)
    snapshot = diffmpc_param_snapshot(mpc)
    history.append(
        {
            "epoch": 0,
            "train_loss_mean": initial_train["loss_mean"],
            "train_loss_max": initial_train["loss_max"],
            "train_rmse_c": initial_train["rmse_c"],
            "val_loss_mean": initial_val["loss_mean"],
            "val_loss_max": initial_val["loss_max"],
            "val_rmse_c": initial_val["rmse_c"],
            **{f"param_{key}": value for key, value in snapshot.items() if isinstance(value, (int, float))},
        }
    )

    for epoch in range(int(epochs)):
        shuffled = np.array(train_indices, copy=True)
        rng.shuffle(shuffled)
        batch_losses = []
        for start in range(0, len(shuffled), int(minibatch_size)):
            mb = shuffled[start : start + int(minibatch_size)]
            if len(mb) == 0:
                continue
            losses = [_sample_loss(int(step_idx)) for step_idx in mb]
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
            opt.step()
            _project_parameters()
            batch_losses.append(float(loss.detach().cpu()))

        train_eval = _evaluate(train_indices)
        val_eval = _evaluate(val_indices)
        score = val_eval["loss_mean"] if len(val_indices) else train_eval["loss_mean"]
        if np.isfinite(score) and score < best_score - 1e-10:
            best_score = score
            best_state = _state_dict_cpu(mpc)
            stale_epochs = 0
        else:
            stale_epochs += 1

        snapshot = diffmpc_param_snapshot(mpc)
        row = {
            "epoch": epoch + 1,
            "batch_loss_mean": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "batch_loss_max": float(np.max(batch_losses)) if batch_losses else float("nan"),
            "train_loss_mean": train_eval["loss_mean"],
            "train_loss_max": train_eval["loss_max"],
            "train_rmse_c": train_eval["rmse_c"],
            "val_loss_mean": val_eval["loss_mean"],
            "val_loss_max": val_eval["loss_max"],
            "val_rmse_c": val_eval["rmse_c"],
            **{f"param_{key}": value for key, value in snapshot.items() if isinstance(value, (int, float))},
        }
        history.append(row)
        print(
            f"Dynamics epoch {epoch + 1:02d}/{int(epochs)} | "
            f"train RMSE {train_eval['rmse_c']:.4f} C | val RMSE {val_eval['rmse_c']:.4f} C"
        )
        if stale_epochs >= int(patience):
            print(f"Dynamics adaptation early-stopped after {epoch + 1} epochs.")
            break

    mpc.load_state_dict(best_state)
    final_train = _evaluate(train_indices)
    final_val = _evaluate(val_indices)
    snapshot = diffmpc_param_snapshot(mpc)
    history.append(
        {
            "epoch": -1,
            "train_loss_mean": final_train["loss_mean"],
            "train_loss_max": final_train["loss_max"],
            "train_rmse_c": final_train["rmse_c"],
            "val_loss_mean": final_val["loss_mean"],
            "val_loss_max": final_val["loss_max"],
            "val_rmse_c": final_val["rmse_c"],
            **{f"param_{key}": value for key, value in snapshot.items() if isinstance(value, (int, float))},
        }
    )
    return history


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
    train_dynamics: bool = False,
) -> list[dict[str, float]]:
    import numpy as np
    import torch

    from src.controllers.rlmpc import (
        diffmpc_param_snapshot,
        freeze_cost_parameters,
        freeze_dynamics_parameters,
        trainable_cost_parameters,
        trainable_dynamics_parameters,
    )

    def _project_dynamics_parameters() -> None:
        with torch.no_grad():
            for raw_parameter in [mpc.C_raw, mpc.Rm_raw, mpc.Rout_raw, mpc.Aeff_raw, mpc.Pnom_raw]:
                raw_parameter.clamp_(-20.0, 20.0)
            mpc.Tm.clamp_(-10.0, 10.0)
            mpc.cop_a.clamp_(float(mpc.cop_min), float(mpc.cop_max))
            mpc.cop_b.clamp_(-2.0, 2.0)

    mpc.train()
    if train_dynamics:
        freeze_cost_parameters(mpc)
        parameters = trainable_dynamics_parameters(mpc)
        update_type = "ppo_dynamics"
    else:
        freeze_dynamics_parameters(mpc)
        parameters = trainable_cost_parameters(mpc)
        update_type = "ppo_cost"

    if not parameters:
        if train_dynamics:
            print("PPO dynamics update skipped: no trainable DiffMPC dynamics parameters.")
        else:
            print("PPO update skipped: RLMPC inner cost is fixed to the matched MPC objective.")
        return []
    opt = torch.optim.Adam(parameters, lr=float(lr))

    n_steps = len(trajectory)
    idxs = np.arange(n_steps)
    logp_old = torch.tensor([step["logp_old"] for step in trajectory], dtype=torch.float32, device=device)
    a_sample = torch.tensor([step["a_sample"] for step in trajectory], dtype=torch.float32, device=device)
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    policy_cls = StochasticDiffMPCPolicy if policy_cls is None else policy_cls
    policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
    policy = policy_cls(mpc, normalization, indices, sigma=sigma, device=device, **policy_kwargs)
    initial_snapshot = diffmpc_param_snapshot(mpc)
    history = [
        {
            "update_type": update_type,
            "epoch": 0,
            "ppo_epoch": 0,
            "loss": float("nan"),
            **{f"param_{key}": value for key, value in initial_snapshot.items() if isinstance(value, (int, float))},
        }
    ]

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
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            opt.step()
            if train_dynamics:
                _project_dynamics_parameters()
            ep_losses.append(float(loss.item()))

        snapshot = diffmpc_param_snapshot(mpc)
        row = {
            "update_type": update_type,
            "epoch": ep + 1,
            "ppo_epoch": ep + 1,
            "loss": float(np.mean(ep_losses)),
            "q_track": float(mpc.q_track().detach().cpu()),
            "r_u": float(mpc.r_u().detach().cpu()),
            "sp_bias": float(mpc.sp_bias().detach().cpu()),
            **{f"param_{key}": value for key, value in snapshot.items() if isinstance(value, (int, float))},
        }
        if hasattr(mpc, "balanced_weight_vector"):
            alpha = mpc.balanced_weight_vector().detach().cpu().numpy()
            row.update(
                {
                    "alpha_track": float(alpha[0]),
                    "alpha_cold": float(alpha[1]),
                    "alpha_hot": float(alpha[2]),
                }
            )
        history.append(row)
        if train_dynamics:
            print(
                f"PPO dynamics epoch {ep + 1:02d}/{int(ppo_epochs)} | loss {row['loss']:.6f} | "
                f"C={row.get('param_C', float('nan')):.4f} "
                f"Rm={row.get('param_Rm', float('nan')):.4f} "
                f"Pnom={row.get('param_Pnom', float('nan')):.4f}"
            )
        elif {"alpha_track", "alpha_cold", "alpha_hot"}.issubset(row):
            print(
                f"PPO epoch {ep + 1:02d}/{int(ppo_epochs)} | loss {row['loss']:.6f} | "
                f"alpha=({row['alpha_track']:.4f}, {row['alpha_cold']:.4f}, {row['alpha_hot']:.4f})"
            )
        else:
            print(
                f"PPO epoch {ep + 1:02d}/{int(ppo_epochs)} | loss {row['loss']:.6f} | "
                f"q_track={row['q_track']:.4f} r_u={row['r_u']:.4f} sp_bias={row['sp_bias']:.4f}"
            )

    mpc.eval()
    return history


def make_raw_env_window(
    *,
    sim_start: int,
    sim_end: int,
    seed: int = 49,
    reward_function=None,
    reward_function_kwargs=None,
    building_name: str = CONTROL_BUILDING_NAME,
):
    from src.envs import EnvConfig, make_citylearn_env

    dataset_dir = Path("data") / "datasets" / DATASET_NAME
    config = EnvConfig(
        schema_path=dataset_dir / "schema.json",
        dataset_dir=dataset_dir,
        buildings=[building_name],
        start_time_step=int(sim_start),
        end_time_step=int(sim_end),
        random_seed=int(seed),
        central_agent=True,
        active_actions=("heating_device",),
    )
    overrides = {}
    if reward_function is not None:
        overrides["reward_function"] = reward_function
    if reward_function_kwargs is not None:
        overrides["reward_function_kwargs"] = reward_function_kwargs
    return make_citylearn_env(config, **overrides)


def make_february_raw_env(
    *,
    seed: int = 49,
    reward_function=None,
    reward_function_kwargs=None,
    building_name: str = CONTROL_BUILDING_NAME,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
):
    return make_raw_env_window(
        sim_start=sim_start,
        sim_end=sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )


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
    reward_function_kwargs=None,
    device,
    max_steps: int | None = None,
    occupant=None,
    policy_cls=None,
    policy_kwargs: dict[str, Any] | None = None,
    label: str = "RLMPC",
    building_name: str = CONTROL_BUILDING_NAME,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
) -> dict[str, Any]:
    import numpy as np

    env = make_february_raw_env(
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
        sim_start=sim_start,
        sim_end=sim_end,
    )
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
    reference_setpoints = []
    net_electricity = []
    heating_electricity = []
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
        if hasattr(policy, "effective_setpoint_profile"):
            policy.effective_setpoint_profile = np.asarray(
                building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point,
                dtype=float,
            )
        action = policy.predict(obs)
        controller_reference = effective
        if getattr(policy, "T_dyn", None) is not None:
            gate_now = float(np.clip(getattr(policy, "last_occ_now", 0.0), 0.0, 1.0))
            controller_reference = (1.0 - gate_now) * scheduled + gate_now * float(policy.T_dyn)
        if getattr(policy, "T_dyn", None) is not None:
            _set_reward_reference_setpoint(env, _policy_reward_reference(env, policy))
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
        reference_setpoints.append(float(controller_reference))
        feedback_hist.append(float(feedback))
        if getattr(policy, "T_dyn", None) is not None:
            tdyn_hist.append(float(getattr(policy, "T_dyn")))
        if hasattr(policy, "last_occ_now"):
            occ_now_hist.append(float(policy.last_occ_now))
            occ_hat_hist.append(float(policy.last_occ_hat))
            occ_bin_hist.append(float(policy.last_occ_bin))
        if hasattr(building, "net_electricity_consumption"):
            net_electricity.append(float(building.net_electricity_consumption[-1]))
        if hasattr(building, "heating_electricity_consumption"):
            heating_electricity.append(float(building.heating_electricity_consumption[-1]))
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
        "building_name": building_name,
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
        "reference_setpoints": np.asarray(reference_setpoints, dtype=float),
        "net_electricity_consumption": np.asarray(net_electricity, dtype=float),
        "heating_electricity_consumption": np.asarray(heating_electricity, dtype=float),
        "feedback_hist": np.asarray(feedback_hist, dtype=float),
        "override_count": int(getattr(env, "override_count", 0)),
        "comfort_band_lower": float(occupant_probability_comfort_band(occupant)[0]),
        "comfort_band_upper": float(occupant_probability_comfort_band(occupant)[1]),
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
    if len(result.get("reference_setpoints", [])) == n_steps:
        data["reference_setpoint"] = np.asarray(result["reference_setpoints"], dtype=float)
    if len(result.get("net_electricity_consumption", [])) == n_steps:
        data["net_electricity_consumption"] = np.asarray(result["net_electricity_consumption"], dtype=float)
    if len(result.get("heating_electricity_consumption", [])) == n_steps:
        data["heating_electricity_consumption"] = np.asarray(result["heating_electricity_consumption"], dtype=float)
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
    comfort_low = float(result.get("comfort_band_lower", 20.0))
    comfort_high = float(result.get("comfort_band_upper", 24.0))
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "override_hours": override_hours(
            result.get("baseline_setpoints", []),
            result.get("effective_setpoints", []),
        ),
        "reward_total": float(result["reward_hist"].sum()) if len(result["reward_hist"]) else 0.0,
        "kelvin_hours": kelvin_hours(result.get("Tin_hist", []), lower=comfort_low, upper=comfort_high),
        "setpoint_kelvin_hours": setpoint_kelvin_hours(
            result.get("Tin_hist", []),
            result.get("effective_setpoints", []),
        ),
        "electricity_consumption_total": _get_citylearn_kpi(result["district_kpis"], "electricity_consumption_total"),
        "discomfort_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_proportion"),
        "discomfort_cold_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_cold_proportion"),
        "discomfort_hot_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_hot_proportion"),
        "daily_peak_average": _get_citylearn_kpi(result["district_kpis"], "daily_peak_average"),
    }
    if online_info:
        summary.update(online_info)
    if "building_name" in result:
        summary["building_name"] = result["building_name"]
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "comfort_band_lower" in result:
        summary["comfort_band_lower"] = float(result["comfort_band_lower"])
    if "comfort_band_upper" in result:
        summary["comfort_band_upper"] = float(result["comfort_band_upper"])
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
    reward_function_kwargs=None,
    device: str = "auto",
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr_ppo: float = 5e-3,
    gamma: float = 0.99,
    max_steps: int | None,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
    label: str = "baseline",
    building_name: str = CONTROL_BUILDING_NAME,
    building_metadata: dict[str, Any] | None = None,
    matched_mpc_cost: dict[str, float] | None = None,
    stress_profile=None,
    future_stress_profile=None,
    setpoint_bias_profile=None,
    district_target_profile=None,
    district_background_profile=None,
    district_action_power_profile=None,
    adaptive_dynamics: bool = False,
    dynamics_epochs: int = 5,
    dynamics_lr: float = 1e-3,
    dynamics_minibatch_size: int = 64,
    dynamics_max_grad_norm: float = 1.0,
    dynamics_validation_fraction: float = 0.2,
    dynamics_patience: int = 3,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from src.controllers.rlmpc import diffmpc_param_snapshot

    set_experiment_seed(training_seed)
    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    if matched_mpc_cost is not None:
        mpc.use_balanced_mpc_cost(
            x_std=float(checkpoint["normalization"]["x_std"]),
            **matched_mpc_cost,
        )
    env_probe = make_february_raw_env(
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
        sim_start=sim_start,
        sim_end=sim_end,
    )
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

    env_train = make_february_raw_env(
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    policy_stoch = StochasticDiffMPCPolicy(
        mpc,
        checkpoint["normalization"],
        env_indices,
        sigma=sigma,
        device=torch_device,
        stress_profile=stress_profile,
        future_stress_profile=future_stress_profile,
        setpoint_bias_profile=setpoint_bias_profile,
        district_target_profile=district_target_profile,
        district_background_profile=district_background_profile,
        district_action_power_profile=district_action_power_profile,
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
    dynamics_history = []
    params_after_dynamics = params_before
    if adaptive_dynamics:
        print("Running online RLMPC PPO dynamics adaptation through the DiffMPC policy.")
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
        train_dynamics=adaptive_dynamics,
        policy_kwargs={
            "stress_profile": stress_profile,
            "future_stress_profile": future_stress_profile,
            "setpoint_bias_profile": setpoint_bias_profile,
            "district_target_profile": district_target_profile,
            "district_background_profile": district_background_profile,
            "district_action_power_profile": district_action_power_profile,
        },
    )
    if adaptive_dynamics:
        dynamics_history = [dict(row) for row in ppo_history]
        params_after_dynamics = diffmpc_param_snapshot(mpc)
    params_after = diffmpc_param_snapshot(mpc)

    result = evaluate_diffmpc_policy(
        mpc=mpc,
        normalization=checkpoint["normalization"],
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        device=torch_device,
        max_steps=max_steps,
        building_name=building_name,
        sim_start=sim_start,
        sim_end=sim_end,
        policy_kwargs={
            "stress_profile": stress_profile,
            "future_stress_profile": future_stress_profile,
            "setpoint_bias_profile": setpoint_bias_profile,
            "district_target_profile": district_target_profile,
            "district_background_profile": district_background_profile,
            "district_action_power_profile": district_action_power_profile,
        },
    )

    online_info = {
        "train_reward_total": float(np.sum([step["reward"] for step in trajectory])),
        "train_steps": int(len(trajectory)),
        "simulation_start": int(sim_start),
        "simulation_end": int(sim_end),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "q_track_before": float(params_before["q_track"]),
        "r_u_before": float(params_before["r_u"]),
        "sp_bias_before": float(params_before["sp_bias"]),
        "q_track_after": float(params_after["q_track"]),
        "r_u_after": float(params_after["r_u"]),
        "sp_bias_after": float(params_after["sp_bias"]),
        "alpha_track_before": float(params_before.get("alpha_track", float("nan"))),
        "alpha_cold_before": float(params_before.get("alpha_cold", float("nan"))),
        "alpha_hot_before": float(params_before.get("alpha_hot", float("nan"))),
        "alpha_track_after": float(params_after.get("alpha_track", float("nan"))),
        "alpha_cold_after": float(params_after.get("alpha_cold", float("nan"))),
        "alpha_hot_after": float(params_after.get("alpha_hot", float("nan"))),
        "balanced_adaptive_weights": bool(params_after.get("balanced_adaptive_weights", False)),
        "adaptive_dynamics": bool(adaptive_dynamics),
        "dynamics_update_type": "ppo_dynamics" if adaptive_dynamics else "none",
        "dynamics_epochs_requested": int(dynamics_epochs),
        "dynamics_lr": float(dynamics_lr),
        "dynamics_minibatch_size": int(dynamics_minibatch_size),
        "dynamics_validation_fraction": float(dynamics_validation_fraction),
        "dynamics_patience": int(dynamics_patience),
    }
    for key in ["C", "Rm", "Rout", "Aeff", "Pnom", "Tm", "cop_a", "cop_b"]:
        online_info[f"{key}_before"] = float(params_before.get(key, float("nan")))
        online_info[f"{key}_after_dynamics"] = float(params_after_dynamics.get(key, float("nan")))
        online_info[f"{key}_after"] = float(params_after.get(key, float("nan")))
    if dynamics_history:
        initial_dyn = dynamics_history[0]
        final_dyn = dynamics_history[-1]
        online_info.update(
            {
                "dynamics_epochs_run": int(max(0, len(dynamics_history) - 1)),
                "dynamics_train_rmse_c_before": float(initial_dyn.get("train_rmse_c", float("nan"))),
                "dynamics_train_rmse_c_after": float(final_dyn.get("train_rmse_c", float("nan"))),
                "dynamics_val_rmse_c_before": float(initial_dyn.get("val_rmse_c", float("nan"))),
                "dynamics_val_rmse_c_after": float(final_dyn.get("val_rmse_c", float("nan"))),
                "dynamics_train_loss_before": float(initial_dyn.get("train_loss_mean", float("nan"))),
                "dynamics_train_loss_after": float(final_dyn.get("train_loss_mean", float("nan"))),
                "dynamics_val_loss_before": float(initial_dyn.get("val_loss_mean", float("nan"))),
                "dynamics_val_loss_after": float(final_dyn.get("val_loss_mean", float("nan"))),
                "dynamics_ppo_loss_before": float(initial_dyn.get("loss", float("nan"))),
                "dynamics_ppo_loss_after": float(final_dyn.get("loss", float("nan"))),
            }
        )
    if matched_mpc_cost is not None:
        online_info.update(
            {
                "mpc_cost_mode": "balanced_mpc",
                "matched_mpc_cost": True,
                "matched_mpc_comfort_weight": float(matched_mpc_cost["comfort_weight"]),
                "matched_mpc_energy_weight": float(matched_mpc_cost["energy_weight"]),
                "matched_mpc_peak_weight": float(matched_mpc_cost["peak_weight"]),
                "matched_mpc_ramp_weight": float(matched_mpc_cost["ramp_weight"]),
                "matched_mpc_shift_weight": float(matched_mpc_cost.get("shift_weight", 0.0)),
                "matched_mpc_preheat_weight": float(matched_mpc_cost.get("preheat_weight", 0.0)),
                "matched_mpc_district_target_weight": float(
                    matched_mpc_cost.get("district_target_weight", 0.0)
                ),
                "matched_mpc_district_target_scale": float(
                    matched_mpc_cost.get("district_target_scale", 1.0)
                ),
                "matched_mpc_district_target_mode": str(
                    matched_mpc_cost.get("district_target_mode", "tracking")
                ),
                "matched_mpc_comfort_band": float(matched_mpc_cost["comfort_band"]),
                "matched_mpc_comfort_scale": float(matched_mpc_cost["comfort_scale"]),
                "matched_mpc_comfort_mode": str(matched_mpc_cost.get("comfort_mode", "band")),
                "matched_mpc_comfort_track_weight": float(
                    matched_mpc_cost.get("comfort_track_weight", float("nan"))
                ),
                "matched_mpc_comfort_cold_weight": float(
                    matched_mpc_cost.get("comfort_cold_weight", float("nan"))
                ),
                "matched_mpc_comfort_hot_weight": float(
                    matched_mpc_cost.get("comfort_hot_weight", float("nan"))
                ),
                "matched_mpc_adaptive_weights": bool(matched_mpc_cost.get("adaptive_weights", False)),
                "matched_mpc_comfort_cold_tolerance": float(
                    matched_mpc_cost.get("comfort_cold_tolerance", float("nan"))
                ),
                "matched_mpc_comfort_hot_tolerance": float(
                    matched_mpc_cost.get("comfort_hot_tolerance", float("nan"))
                ),
                "matched_mpc_comfort_track_scale": float(
                    matched_mpc_cost.get("comfort_track_scale", float("nan"))
                ),
                "matched_mpc_comfort_cold_scale": float(
                    matched_mpc_cost.get("comfort_cold_scale", float("nan"))
                ),
                "matched_mpc_comfort_hot_scale": float(
                    matched_mpc_cost.get("comfort_hot_scale", float("nan"))
                ),
                "matched_mpc_time_bias_guard_band": float(
                    matched_mpc_cost.get("time_bias_guard_band", float("nan"))
                ),
                "matched_mpc_power_scale": float(matched_mpc_cost["power_scale"]),
                "matched_mpc_power_per_unit_action": float(matched_mpc_cost["power_per_unit_action"]),
                "matched_mpc_normalize_terms": bool(matched_mpc_cost["normalize_terms"]),
            }
        )
    if building_metadata:
        online_info.update(building_metadata)
    if stress_profile is not None:
        online_info["grid_stress_profile_length"] = int(len(stress_profile))
        online_info["grid_stress_profile_mean"] = float(np.mean(stress_profile)) if len(stress_profile) else 0.0
    if future_stress_profile is not None:
        online_info["future_grid_stress_profile_length"] = int(len(future_stress_profile))
        online_info["future_grid_stress_profile_mean"] = (
            float(np.mean(future_stress_profile)) if len(future_stress_profile) else 0.0
        )
    if setpoint_bias_profile is not None:
        online_info["setpoint_bias_profile_length"] = int(len(setpoint_bias_profile))
        online_info["setpoint_bias_profile_mean"] = (
            float(np.mean(setpoint_bias_profile)) if len(setpoint_bias_profile) else 0.0
        )
    if district_target_profile is not None:
        online_info["district_target_profile_length"] = int(len(district_target_profile))
        online_info["district_target_mean"] = float(np.mean(district_target_profile)) if len(district_target_profile) else 0.0
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
    pd.DataFrame(dynamics_history).to_csv(output_case_dir / "dynamics_history.csv", index=False)
    updated_checkpoint = {
        **checkpoint,
        "state_dict": {key: value.detach().cpu().clone() for key, value in mpc.state_dict().items()},
        "online_info": online_info,
        "ppo_history": ppo_history,
        "dynamics_history": dynamics_history,
    }
    online_checkpoint_path = output_case_dir / "online_updated_checkpoint.pt"
    torch.save(updated_checkpoint, online_checkpoint_path)

    summary_path = Path(summary_dir) / "no_occupant" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([metrics])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if "case" in existing.columns and "case" in new_row.columns:
            existing = existing[existing["case"] != metrics["case"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(summary_path, index=False)

    print("February reward after PPO:", metrics["reward_total"])
    print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    print(f"Saved rollout to {output_case_dir / 'rollout.csv'}")
    print(f"Saved summary to {summary_path}")
    return {
        "result": result,
        "metrics": metrics,
        "ppo_history": ppo_history,
        "dynamics_history": dynamics_history,
    }


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
    reward_function_kwargs: dict[str, Any] | None,
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
    sim_start: int,
    sim_end: int,
    building_name: str,
    occupancy_building_name: str,
    building_metadata: dict[str, Any] | None,
    adaptive_dynamics: bool = False,
    dynamics_epochs: int = 5,
    dynamics_lr: float = 1e-3,
    dynamics_minibatch_size: int = 64,
    dynamics_max_grad_norm: float = 1.0,
    dynamics_validation_fraction: float = 0.2,
    dynamics_patience: int = 3,
    matched_mpc_cost: dict[str, float] | None = None,
    stress_profile=None,
    future_stress_profile=None,
    setpoint_bias_profile=None,
    district_target_profile=None,
    district_background_profile=None,
    district_action_power_profile=None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from src.controllers.rlmpc import diffmpc_param_snapshot

    set_experiment_seed(training_seed)
    mpc, checkpoint, torch_device = load_offline_checkpoint(checkpoint_path, device=device)
    if matched_mpc_cost is not None:
        mpc.use_balanced_mpc_cost(
            x_std=float(checkpoint["normalization"]["x_std"]),
            **matched_mpc_cost,
        )
    env_probe = make_february_raw_env(
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
        sim_start=sim_start,
        sim_end=sim_end,
    )
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
        occupancy_model, occupancy_series = fit_bspline_occupancy_model(
            seed=seed,
            building_name=occupancy_building_name,
        )

    params_before = diffmpc_param_snapshot(mpc)
    param_history = [{"stage": "offline_init", "week": 0, **params_before}]
    weekly_training = []
    ppo_rows = []
    dynamics_rows = []
    n_weeks = int(np.ceil((int(sim_end) - int(sim_start) + 1) / WEEK_HOURS))

    for week_idx in range(n_weeks):
        week_start = int(sim_start) + week_idx * WEEK_HOURS
        week_end = min(week_start + WEEK_HOURS - 1, int(sim_end))
        week_seed = int(seed + week_idx)
        env_train = make_raw_env_window(
            sim_start=week_start,
            sim_end=week_end,
            seed=week_seed,
            reward_function=reward_function,
            reward_function_kwargs=reward_function_kwargs,
            building_name=building_name,
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
                stress_profile=stress_profile,
                future_stress_profile=future_stress_profile,
                setpoint_bias_profile=setpoint_bias_profile,
                district_target_profile=district_target_profile,
                district_background_profile=district_background_profile,
                district_action_power_profile=district_action_power_profile,
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
                "stress_profile": stress_profile,
                "future_stress_profile": future_stress_profile,
                "setpoint_bias_profile": setpoint_bias_profile,
                "district_target_profile": district_target_profile,
                "district_background_profile": district_background_profile,
                "district_action_power_profile": district_action_power_profile,
            }
        else:
            policy_stoch = StochasticDiffMPCPolicy(
                mpc,
                checkpoint["normalization"],
                env_indices,
                sigma=sigma,
                device=torch_device,
                stress_profile=stress_profile,
                future_stress_profile=future_stress_profile,
                setpoint_bias_profile=setpoint_bias_profile,
                district_target_profile=district_target_profile,
                district_background_profile=district_background_profile,
                district_action_power_profile=district_action_power_profile,
            )
            ppo_policy_cls = StochasticDiffMPCPolicy
            ppo_policy_kwargs = {
                "stress_profile": stress_profile,
                "future_stress_profile": future_stress_profile,
                "setpoint_bias_profile": setpoint_bias_profile,
                "district_target_profile": district_target_profile,
                "district_background_profile": district_background_profile,
                "district_action_power_profile": district_action_power_profile,
            }

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
            train_dynamics=adaptive_dynamics,
        )
        for row in ppo_history:
            row_with_week = {"week": week_idx + 1, **row}
            ppo_rows.append(row_with_week)
            if adaptive_dynamics:
                dynamics_rows.append(row_with_week)

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
            "sim_start": int(sim_start),
            "T_dyn_init_mode": T_dyn_init_mode,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "drift_to_pref": drift_to_pref,
            "T_dyn_min": T_dyn_min,
            "T_dyn_max": T_dyn_max,
            "stress_profile": stress_profile,
            "future_stress_profile": future_stress_profile,
            "setpoint_bias_profile": setpoint_bias_profile,
            "district_target_profile": district_target_profile,
            "district_background_profile": district_background_profile,
            "district_action_power_profile": district_action_power_profile,
        }
    else:
        eval_policy_cls = DiffMPCPolicyRaw
        eval_policy_kwargs = {
            "stress_profile": stress_profile,
            "future_stress_profile": future_stress_profile,
            "setpoint_bias_profile": setpoint_bias_profile,
            "district_target_profile": district_target_profile,
            "district_background_profile": district_background_profile,
            "district_action_power_profile": district_action_power_profile,
        }

    result = evaluate_diffmpc_policy(
        mpc=mpc,
        normalization=checkpoint["normalization"],
        seed=eval_seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        device=torch_device,
        max_steps=max_steps,
        occupant=deepcopy(occupant),
        policy_cls=eval_policy_cls,
        policy_kwargs=eval_policy_kwargs,
        label=label,
        building_name=building_name,
        sim_start=int(sim_start),
        sim_end=int(sim_end),
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
        "alpha_track_before": float(params_before.get("alpha_track", float("nan"))),
        "alpha_cold_before": float(params_before.get("alpha_cold", float("nan"))),
        "alpha_hot_before": float(params_before.get("alpha_hot", float("nan"))),
        "alpha_track_after": float(params_after.get("alpha_track", float("nan"))),
        "alpha_cold_after": float(params_after.get("alpha_cold", float("nan"))),
        "alpha_hot_after": float(params_after.get("alpha_hot", float("nan"))),
        "balanced_adaptive_weights": bool(params_after.get("balanced_adaptive_weights", False)),
        "adaptive_dynamics": bool(adaptive_dynamics),
        "dynamics_update_type": "ppo_dynamics" if adaptive_dynamics else "none",
        "dynamics_epochs_requested": int(dynamics_epochs),
        "dynamics_lr": float(dynamics_lr),
        "dynamics_minibatch_size": int(dynamics_minibatch_size),
        "dynamics_validation_fraction": float(dynamics_validation_fraction),
        "dynamics_patience": int(dynamics_patience),
    }
    for key in ["C", "Rm", "Rout", "Aeff", "Pnom", "Tm", "cop_a", "cop_b"]:
        online_info[f"{key}_before"] = float(params_before.get(key, float("nan")))
        online_info[f"{key}_after"] = float(params_after.get(key, float("nan")))
    if dynamics_rows:
        initial_dyn = dynamics_rows[0]
        final_dyn = dynamics_rows[-1]
        online_info.update(
            {
                "dynamics_epochs_run": int(sum(1 for row in dynamics_rows if int(row.get("ppo_epoch", 0)) > 0)),
                "dynamics_train_rmse_c_before": float(initial_dyn.get("train_rmse_c", float("nan"))),
                "dynamics_train_rmse_c_after": float(final_dyn.get("train_rmse_c", float("nan"))),
                "dynamics_val_rmse_c_before": float(initial_dyn.get("val_rmse_c", float("nan"))),
                "dynamics_val_rmse_c_after": float(final_dyn.get("val_rmse_c", float("nan"))),
                "dynamics_train_loss_before": float(initial_dyn.get("train_loss_mean", float("nan"))),
                "dynamics_train_loss_after": float(final_dyn.get("train_loss_mean", float("nan"))),
                "dynamics_val_loss_before": float(initial_dyn.get("val_loss_mean", float("nan"))),
                "dynamics_val_loss_after": float(final_dyn.get("val_loss_mean", float("nan"))),
                "dynamics_ppo_loss_before": float(initial_dyn.get("loss", float("nan"))),
                "dynamics_ppo_loss_after": float(final_dyn.get("loss", float("nan"))),
            }
        )
    if matched_mpc_cost is not None:
        online_info.update(
            {
                "mpc_cost_mode": "balanced_mpc",
                "matched_mpc_cost": True,
                "matched_mpc_comfort_weight": float(matched_mpc_cost["comfort_weight"]),
                "matched_mpc_energy_weight": float(matched_mpc_cost["energy_weight"]),
                "matched_mpc_peak_weight": float(matched_mpc_cost["peak_weight"]),
                "matched_mpc_ramp_weight": float(matched_mpc_cost["ramp_weight"]),
                "matched_mpc_shift_weight": float(matched_mpc_cost.get("shift_weight", 0.0)),
                "matched_mpc_preheat_weight": float(matched_mpc_cost.get("preheat_weight", 0.0)),
                "matched_mpc_comfort_mode": str(matched_mpc_cost.get("comfort_mode", "band")),
                "matched_mpc_comfort_track_weight": float(matched_mpc_cost.get("comfort_track_weight", float("nan"))),
                "matched_mpc_comfort_cold_weight": float(matched_mpc_cost.get("comfort_cold_weight", float("nan"))),
                "matched_mpc_comfort_hot_weight": float(matched_mpc_cost.get("comfort_hot_weight", float("nan"))),
                "matched_mpc_adaptive_weights": bool(matched_mpc_cost.get("adaptive_weights", False)),
                "matched_mpc_comfort_track_scale": float(matched_mpc_cost.get("comfort_track_scale", float("nan"))),
                "matched_mpc_comfort_cold_scale": float(matched_mpc_cost.get("comfort_cold_scale", float("nan"))),
                "matched_mpc_comfort_hot_scale": float(matched_mpc_cost.get("comfort_hot_scale", float("nan"))),
                "matched_mpc_time_bias_guard_band": float(matched_mpc_cost.get("time_bias_guard_band", float("nan"))),
                "matched_mpc_power_scale": float(matched_mpc_cost["power_scale"]),
                "matched_mpc_power_per_unit_action": float(matched_mpc_cost["power_per_unit_action"]),
                "matched_mpc_normalize_terms": bool(matched_mpc_cost["normalize_terms"]),
            }
        )
    if building_metadata:
        online_info.update(building_metadata)
    if setpoint_bias_profile is not None:
        online_info["setpoint_bias_profile_length"] = int(len(setpoint_bias_profile))
        online_info["setpoint_bias_profile_mean"] = (
            float(np.mean(setpoint_bias_profile)) if len(setpoint_bias_profile) else 0.0
        )
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
    pd.DataFrame(dynamics_rows).to_csv(output_case_dir / "dynamics_history.csv", index=False)
    pd.DataFrame(weekly_training).to_csv(output_case_dir / "weekly_training.csv", index=False)
    pd.DataFrame(param_history).to_csv(output_case_dir / "param_history.csv", index=False)
    updated_checkpoint = {
        **checkpoint,
        "state_dict": {key: value.detach().cpu().clone() for key, value in mpc.state_dict().items()},
        "online_info": online_info,
        "ppo_history": ppo_rows,
        "dynamics_history": dynamics_rows,
        "weekly_training": weekly_training,
        "param_history": param_history,
    }
    torch.save(updated_checkpoint, output_case_dir / "online_updated_checkpoint.pt")

    print("Reward:", metrics["reward_total"])
    print("Overrides:", metrics["override_count"])
    if mode == "bspline_tdyn" and "tdyn_final" in metrics:
        print(f"T_dyn mean/final: {metrics['tdyn_mean']:.3f} / {metrics['tdyn_final']:.3f}")
    print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return {"result": result, "metrics": metrics, "ppo_history": ppo_rows, "dynamics_history": dynamics_rows}


def run_online_rlmpc_occupants(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path = "results/raw/rlmpc",
    summary_dir: str | Path = "results/summaries/rlmpc",
    seed: int = 49,
    training_seed: int = 0,
    reward_function=None,
    reward_function_kwargs: dict[str, Any] | None = None,
    device: str = "auto",
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr_ppo: float = 5e-3,
    gamma: float = 0.99,
    adaptive_dynamics: bool = False,
    dynamics_epochs: int = 5,
    dynamics_lr: float = 1e-3,
    dynamics_minibatch_size: int = 64,
    dynamics_max_grad_norm: float = 1.0,
    dynamics_validation_fraction: float = 0.2,
    dynamics_patience: int = 3,
    max_steps: int | None = None,
    label_prefix: str = "",
    occupant: str = "all",
    occupant_mode: str = "both",
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.05,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
    building_name: str = CONTROL_BUILDING_NAME,
    occupancy_building_name: str | None = None,
    building_metadata: dict[str, Any] | None = None,
    matched_mpc_cost: dict[str, float] | None = None,
    stress_profile=None,
    future_stress_profile=None,
    setpoint_bias_profile=None,
    district_target_profile=None,
    district_background_profile=None,
    district_action_power_profile=None,
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
    occupancy_building_name = building_name if occupancy_building_name is None else occupancy_building_name
    for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
        for mode in mode_names:
            label = f"{label_prefix}_{name}_{mode}" if label_prefix else f"{name}_{mode}"
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
                reward_function_kwargs=reward_function_kwargs,
                device=device,
                sigma=sigma,
                clip_eps=clip_eps,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                lr_ppo=lr_ppo,
                gamma=gamma,
                adaptive_dynamics=adaptive_dynamics,
                dynamics_epochs=dynamics_epochs,
                dynamics_lr=dynamics_lr,
                dynamics_minibatch_size=dynamics_minibatch_size,
                dynamics_max_grad_norm=dynamics_max_grad_norm,
                dynamics_validation_fraction=dynamics_validation_fraction,
                dynamics_patience=dynamics_patience,
                max_steps=max_steps,
                alpha_occ=alpha_occ,
                T_dyn_init_mode=T_dyn_init_mode,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                T_dyn_min=T_dyn_min,
                T_dyn_max=T_dyn_max,
                sim_start=int(sim_start),
                sim_end=int(sim_end),
                building_name=building_name,
                occupancy_building_name=occupancy_building_name,
                building_metadata=building_metadata,
                matched_mpc_cost=matched_mpc_cost,
                stress_profile=stress_profile,
                future_stress_profile=future_stress_profile,
                setpoint_bias_profile=setpoint_bias_profile,
                district_target_profile=district_target_profile,
                district_background_profile=district_background_profile,
                district_action_power_profile=district_action_power_profile,
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


# ---------------------------------------------------------------------------
# 2R2C DiffMPC — offline imitation learning + online PPO
# ---------------------------------------------------------------------------

def _estimate_tm_sequence(
    states_np,
    actions_np,
    indices: dict[str, Any],
    rc2_params: dict[str, float],
    dones_np=None,
) -> "np.ndarray":
    """Roll out the 2R2C model on January PI data to estimate Tm at each step.

    Returns tm_seq of shape (N,) where tm_seq[t] is the mass temperature at step t.
    """
    import numpy as np

    N = states_np.shape[0]
    cop_a = float(rc2_params["cop_a"])
    cop_b = float(rc2_params["cop_b"])
    a11 = float(rc2_params["a11"])
    a12 = float(rc2_params["a12"])
    a21 = float(rc2_params["a21"])
    a22 = float(rc2_params["a22"])
    b_tout = float(rc2_params["b_tout"])
    b_sol = float(rc2_params["b_sol"])
    b_heat = float(rc2_params["b_heat"])
    b0 = float(rc2_params["b0"])

    tin_idx = indices["tin"]
    tout_idx = indices["tout"]
    sol_idx = indices["solar"]

    tm_seq = np.zeros(N, dtype=np.float32)
    done_arr = None if dones_np is None else np.asarray(dones_np).reshape(-1).astype(bool)
    tm = float(states_np[0, tin_idx])  # initialise Tm = Tin at t=0
    for t in range(N):
        if t > 0 and done_arr is not None and done_arr[t - 1]:
            tm = float(states_np[t, tin_idx])
        tm_seq[t] = tm
        tin = float(states_np[t, tin_idx])
        tout = float(states_np[t, tout_idx])
        sol = float(states_np[t, sol_idx])
        u = float(actions_np[t, 0]) if actions_np.ndim == 2 else float(actions_np[t])
        cop = float(np.clip(cop_a + cop_b * tout, 0.5, 6.0))
        tin_next = a11 * tin + a12 * tm + b_tout * tout + b_sol * sol + b_heat * cop * u + b0
        tm_next = a21 * tin + a22 * tm
        tm = tm_next
        _ = tin_next  # predicted but we use observed Tin from data
    return tm_seq


def train_offline_2r2c_diffmpc(
    *,
    dataset_path: str | Path,
    rc2_params: dict[str, float],
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    horizon: int = 6,
    eta: float = 10.0,
    rho_u: float = 1.0,
    u_min: float = 0.0,
    u_max: float = 1.0,
    q_reg: float = 1e-4,
    qp_max_iter: int = 200,
    qp_eps: float = 1e-6,
    epochs: int = 200,
    lr: float = 1e-2,
    lambda_dyn: float = 2.0,
    batch_size: int = 32,
    patience: int = 7,
    seed: int = 0,
    device: str = "auto",
    max_samples: int | None = None,
    observation_preset: str = "auto",
    building_name: str | None = None,
    building_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Imitation learning for TwoStateDiffMPC with fixed 2R2C dynamics.

    Learns cost weights (q_track, r_u) by imitating PI actions, with an
    optional one-step Tin prediction loss weighted by lambda_dyn.
    """
    import numpy as np
    import torch

    from src.controllers.rlmpc import TwoStateDiffMPC, twostatediffmpc_param_snapshot

    set_experiment_seed(seed)
    torch_device = resolve_device(device)

    data = np.load(dataset_path)
    states_np = data["states"][:, 0, :].astype(np.float32)
    actions_np = data["actions"][:, 0, :].astype(np.float32)
    next_states_np = data["next_states"][:, 0, :].astype(np.float32)
    dones_np = data["dones"].astype(bool) if "dones" in data.files else None

    if max_samples is not None:
        states_np = states_np[: int(max_samples)]
        actions_np = actions_np[: int(max_samples)]
        next_states_np = next_states_np[: int(max_samples)]
        if dones_np is not None:
            dones_np = dones_np[: int(max_samples)]

    if states_np.shape[0] < 4:
        raise RuntimeError("Need at least four samples for train/validation split.")

    observation_source, observation_names = observation_names_for_dataset(states_np.shape[-1], observation_preset)
    indices = observation_indices_from_names(observation_names)

    # Estimate mass temperature sequence from 2R2C rollout
    tm_seq = _estimate_tm_sequence(
        states_np,
        actions_np[:, 0] if actions_np.ndim == 3 else actions_np,
        indices,
        rc2_params,
        dones_np=dones_np,
    )

    states = torch.tensor(states_np, device=torch_device)
    actions = torch.tensor(actions_np, device=torch_device)
    next_states = torch.tensor(next_states_np, device=torch_device)
    tm_tensor = torch.tensor(tm_seq, device=torch_device)

    n_samples = int(states.shape[0])
    split = max(1, min(n_samples - 1, int(0.8 * n_samples)))
    train_ids = np.arange(0, split)
    val_ids = np.arange(split, n_samples)

    mpc = TwoStateDiffMPC(
        a11=rc2_params["a11"],
        a12=rc2_params["a12"],
        a21=rc2_params["a21"],
        a22=rc2_params["a22"],
        b_tout=rc2_params["b_tout"],
        b_sol=rc2_params["b_sol"],
        b_heat=rc2_params["b_heat"],
        b0=rc2_params["b0"],
        cop_a=rc2_params["cop_a"],
        cop_b=rc2_params["cop_b"],
        T=horizon,
        u_min=u_min,
        u_max=u_max,
        q_reg=q_reg,
        qp_max_iter=qp_max_iter,
        qp_eps=qp_eps,
        eta=eta,
        rho_u=rho_u,
    ).to(torch_device)
    opt = torch.optim.Adam(mpc.parameters(), lr=lr)

    def _d_forecast_2r2c(t: int):
        n = states.shape[0]
        end = min(t + horizon, n)
        d_seq = torch.stack([states[t:end, indices["tout"]], states[t:end, indices["solar"]]], dim=1)
        if d_seq.shape[0] < horizon:
            d_seq = torch.cat([d_seq, d_seq[-1:].repeat(horizon - d_seq.shape[0], 1)], dim=0)
        return d_seq  # (horizon, 2) in physical units

    def _sp_forecast_2r2c(t: int):
        n = states.shape[0]
        end = min(t + horizon, n)
        sp = states[t:end, indices["tsp"]].view(-1, 1)
        if sp.shape[0] < horizon:
            sp = torch.cat([sp, sp[-1:].repeat(horizon - sp.shape[0], 1)], dim=0)
        return sp  # (horizon, 1) in physical °C

    def _one_step_pred_2r2c(tin, tm, d0, u):
        """Predict Tin(t+1) from 2R2C dynamics (all physical units)."""
        tout0 = d0[0, 0]
        sol0 = d0[0, 1]
        cop = torch.clamp(mpc._cop_a + mpc._cop_b * tout0, mpc.cop_min, mpc.cop_max)
        tin_next = (
            mpc.a11 * tin + mpc.a12 * tm
            + mpc.b_tout * tout0 + mpc.b_sol * sol0
            + mpc.b_heat * cop * u + mpc.b0
        )
        return tin_next

    @torch.no_grad()
    def eval_loss(ids) -> float:
        mpc.eval()
        losses = []
        for t in ids:
            tin_t = states[t, indices["tin"]].view(1, 1)
            tm_t = tm_tensor[t].view(1, 1)
            d_seq = _d_forecast_2r2c(int(t)).unsqueeze(0)   # (1, T, 2)
            sp_seq = _sp_forecast_2r2c(int(t)).unsqueeze(0)  # (1, T, 1)
            u_t = actions[t, 0].view(1, 1) if actions.ndim == 3 else actions[t].view(1, 1)
            u_opt, _, _ = mpc(tin_t, tm_t, d_seq, sp_seq)
            tin_next_true = next_states[t, indices["tin"]].view(1)
            tin_next_pred = _one_step_pred_2r2c(tin_t.squeeze(), tm_t.squeeze(), d_seq.squeeze(0), u_t.squeeze())
            loss = lambda_dyn * (tin_next_true - tin_next_pred).pow(2).mean() + (u_t.squeeze() - u_opt).pow(2).mean()
            losses.append(float(loss.item()))
        return float(np.mean(losses))

    best_val = float("inf")
    best_sd = None
    bad_epochs = 0
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []

    print(f"Training samples: {n_samples} | Observation source: {observation_source}")
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
                tin_t = states[t, indices["tin"]].view(1, 1)
                tm_t = tm_tensor[t].view(1, 1)
                d_seq = _d_forecast_2r2c(int(t)).unsqueeze(0)
                sp_seq = _sp_forecast_2r2c(int(t)).unsqueeze(0)
                u_t = actions[t, 0].view(1, 1) if actions.ndim == 3 else actions[t].view(1, 1)
                u_opt, _, _ = mpc(tin_t, tm_t, d_seq, sp_seq)
                tin_next_true = next_states[t, indices["tin"]].view(1)
                tin_next_pred = _one_step_pred_2r2c(tin_t.squeeze(), tm_t.squeeze(), d_seq.squeeze(0), u_t.squeeze())
                loss = lambda_dyn * (tin_next_true - tin_next_pred).pow(2).mean() + (u_t.squeeze() - u_opt).pow(2).mean()
                loss_batch = loss_batch + loss

            loss_batch = loss_batch / max(1, len(batch))
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(mpc.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss_batch.item()))

        val = eval_loss(val_ids)
        train = float(np.mean(batch_losses))
        params = twostatediffmpc_param_snapshot(mpc)
        history.append({"epoch": epoch, "train_loss": train, "val_loss": val, **params})
        print(
            f"Epoch {epoch:03d} | Train {train:.6f} | Val {val:.6f} | "
            f"q_track={params['q_track']:.3g} r_u={params['r_u']:.3g}"
        )

        if val < best_val - 1e-6:
            best_val = val
            best_sd = {k: v.detach().cpu().clone() for k, v in mpc.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                print("Early stopping.")
                break

    if best_sd is None:
        raise RuntimeError("2R2C offline training did not produce a checkpoint.")

    mpc.load_state_dict(best_sd)
    final_params = twostatediffmpc_param_snapshot(mpc)

    # Normalization stats (for policy compatibility)
    normalization = {
        "x_mean": float(states[train_ids, indices["tin"]].mean()),
        "x_std": float(states[train_ids, indices["tin"]].std() + 1e-6),
        "d1_mean": float(states[train_ids, indices["tout"]].mean()),
        "d1_std": float(states[train_ids, indices["tout"]].std() + 1e-6),
        "d2_mean": float(states[train_ids, indices["solar"]].mean()),
        "d2_std": float(states[train_ids, indices["solar"]].std() + 1e-6),
    }

    checkpoint = {
        "state_dict": best_sd,
        "best_val": float(best_val),
        "model_type": "2r2c",
        "rc2_params": {k: float(v) for k, v in rc2_params.items()},
        "model_config": {
            "horizon": int(horizon),
            "u_min": float(u_min),
            "u_max": float(u_max),
            "q_reg": float(q_reg),
            "qp_max_iter": int(qp_max_iter),
            "qp_eps": float(qp_eps),
        },
        "training_config": {
            "epochs": int(epochs),
            "lr": float(lr),
            "lambda_dyn": float(lambda_dyn),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "seed": int(seed),
            "max_samples": None if max_samples is None else int(max_samples),
        },
        "normalization": normalization,
        "indices": indices,
        "observation_source": observation_source,
        "observation_names": observation_names,
        "dataset_path": str(dataset_path),
        "history": history,
        "final_params": final_params,
    }
    if building_name is not None:
        checkpoint["building_name"] = str(building_name)
    if building_metadata:
        checkpoint.update(building_metadata)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    from copy import deepcopy
    bundle = deepcopy(checkpoint)
    bundle["checkpoint_path"] = str(checkpoint_path)
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    bundle_path.write_text(_json.dumps(
        {k: v for k, v in bundle.items() if isinstance(v, (str, int, float, bool, dict, list, type(None)))},
        default=str, indent=2
    ))

    print(f"Saved 2R2C checkpoint: {checkpoint_path}")
    return checkpoint


def load_offline_2r2c_checkpoint(checkpoint_path: str | Path, *, device: str = "auto"):
    """Load a TwoStateDiffMPC offline checkpoint."""
    import torch

    from src.controllers.rlmpc import TwoStateDiffMPC

    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    if checkpoint.get("model_type") != "2r2c":
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a 2R2C checkpoint.")

    rc2 = checkpoint["rc2_params"]
    cfg = checkpoint["model_config"]
    mpc = TwoStateDiffMPC(
        a11=rc2["a11"], a12=rc2["a12"], a21=rc2["a21"], a22=rc2["a22"],
        b_tout=rc2["b_tout"], b_sol=rc2["b_sol"], b_heat=rc2["b_heat"],
        b0=rc2["b0"], cop_a=rc2["cop_a"], cop_b=rc2["cop_b"],
        T=cfg["horizon"],
        u_min=cfg["u_min"], u_max=cfg["u_max"],
        q_reg=cfg["q_reg"], qp_max_iter=cfg["qp_max_iter"], qp_eps=cfg["qp_eps"],
    ).to(torch_device)
    mpc.load_state_dict(checkpoint["state_dict"], strict=False)
    mpc.eval()
    return mpc, checkpoint, torch_device


class TwoStateDiffMPCPolicy:
    """Policy wrapper for TwoStateDiffMPC that maintains the latent Tm state."""

    def __init__(
        self,
        mpc,
        indices: dict[str, Any],
        *,
        device,
        stress_profile=None,
        setpoint_profile=None,
        setpoint_bias_profile=None,
        sigma: float = 0.0,
        observer: str = "propagate",
        kalman_process_var: float = 0.05,
        kalman_measurement_var: float = 0.25,
        kalman_initial_var: float = 4.0,
    ):
        self.mpc = mpc
        self.indices = indices
        self.device = device
        self.stress_profile = None if stress_profile is None else [float(v) for v in stress_profile]
        self.setpoint_profile = None if setpoint_profile is None else [float(v) for v in setpoint_profile]
        self.setpoint_bias_profile = (
            None if setpoint_bias_profile is None else [float(v) for v in setpoint_bias_profile]
        )
        self.sigma = float(sigma)
        self.observer = str(observer).strip().lower()
        if self.observer not in {"propagate", "kalman"}:
            raise ValueError(f"Unknown 2R2C observer: {observer}")
        self.kalman_process_var = max(float(kalman_process_var), 1e-12)
        self.kalman_measurement_var = max(float(kalman_measurement_var), 1e-12)
        self.kalman_initial_var = max(float(kalman_initial_var), 1e-12)
        self.residual_gain = 0.2
        self._step_count = 0
        self._tm = None  # reset on first predict
        self._tm_var = self.kalman_initial_var
        self._prev_state = None
        self._w_est = 0.0

    def reset(self):
        self._tm = None
        self._tm_var = self.kalman_initial_var
        self._prev_state = None
        self._step_count = 0
        self._w_est = 0.0

    def _kalman_update_from_previous(self, tin_now: float) -> None:
        """Correct latent Tm using the newest Tin measurement."""
        import numpy as np

        prev = self._prev_state
        if prev is None:
            return

        def scalar(value):
            return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)

        a11 = scalar(self.mpc.a11)
        a12 = scalar(self.mpc.a12)
        a21 = scalar(self.mpc.a21)
        a22 = scalar(self.mpc.a22)
        b_tout = scalar(self.mpc.b_tout)
        b_sol = scalar(self.mpc.b_sol)
        b_heat = scalar(self.mpc.b_heat)
        b0 = scalar(self.mpc.b0)
        cop = float(
            np.clip(
                scalar(self.mpc._cop_a) + scalar(self.mpc._cop_b) * prev["tout"],
                scalar(self.mpc.cop_min),
                scalar(self.mpc.cop_max),
            )
        )
        tin_pred = (
            a11 * prev["tin"]
            + a12 * prev["tm"]
            + b_tout * prev["tout"]
            + b_sol * prev["solar"]
            + b_heat * cop * prev["u"]
            + b0
            + prev.get("w_est", 0.0)
        )
        innovation = tin_now - tin_pred
        s_var = (a12 * a12 * self._tm_var) + self.kalman_measurement_var
        gain = (self._tm_var * a12) / max(s_var, 1e-12)
        tm_post = prev["tm"] + gain * innovation
        tm_var_post = max((1.0 - gain * a12) * self._tm_var, 1e-12)
        self._tm = a21 * prev["tin"] + a22 * tm_post
        self._tm_var = (a22 * a22 * tm_var_post) + self.kalman_process_var
        self._w_est = prev.get("w_est", 0.0) + self.residual_gain * innovation
        self.mpc.reset_mass_temperature(self._tm)

    def predict(self, obs):
        import numpy as np
        import torch

        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        idx = self.indices
        tin = float(o[idx["tin"]])
        tsp = float(o[idx["tsp"]])
        tout_now = float(o[idx["tout"]])
        sol_now = float(o[idx["solar"]])

        if self._tm is None:
            self._tm = tin  # initialise Tm = Tin at episode start
            self._tm_var = self.kalman_initial_var
            self.mpc.reset_mass_temperature(tin)
        elif self.observer == "kalman":
            self._kalman_update_from_previous(tin)

        T = self.mpc.T
        tout_p = [o[p] for p in idx["tout_p"]]
        sol_p = [o[p] for p in idx["solar_p"]]
        tout_h = [tout_now] + tout_p
        sol_h = [sol_now] + sol_p
        tout_h = [tout_h[min(i, len(tout_h) - 1)] for i in range(T)]
        sol_h = [sol_h[min(i, len(sol_h) - 1)] for i in range(T)]

        d_seq = torch.tensor(
            [[tout_h[i], sol_h[i]] for i in range(T)],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, T, 2)
        if self.setpoint_profile:
            sp_vals = [
                self.setpoint_profile[min(self._step_count + i, len(self.setpoint_profile) - 1)]
                for i in range(T)
            ]
            sp_seq = torch.tensor(sp_vals, dtype=torch.float32, device=self.device).view(1, T, 1)
        else:
            sp_seq = torch.full((1, T, 1), tsp, dtype=torch.float32, device=self.device)
        tin_t = torch.tensor([[tin]], dtype=torch.float32, device=self.device)
        tm_t = torch.tensor([[self._tm]], dtype=torch.float32, device=self.device)

        stress_seq = None
        if self.stress_profile:
            vals = [
                self.stress_profile[min(self._step_count + i, len(self.stress_profile) - 1)]
                for i in range(T)
            ]
            stress_seq = torch.tensor(vals, dtype=torch.float32, device=self.device).view(1, T)
        setpoint_bias_seq = None
        if self.setpoint_bias_profile:
            vals = [
                self.setpoint_bias_profile[min(self._step_count + i, len(self.setpoint_bias_profile) - 1)]
                for i in range(T)
            ]
            setpoint_bias_seq = torch.tensor(vals, dtype=torch.float32, device=self.device).view(1, T)
        w_seq = torch.full((1, T), float(self._w_est), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u_opt, _, _ = self.mpc(
                tin_t,
                tm_t,
                d_seq,
                sp_seq,
                stress_seq,
                setpoint_bias_seq=setpoint_bias_seq,
                w_seq=w_seq,
            )
        u = float(u_opt.item())

        if self.sigma > 0.0:
            u = float(np.clip(u + np.random.normal(0, self.sigma), 0.0, 1.0))

        if self.observer == "kalman":
            self._prev_state = {
                "tin": tin,
                "tm": float(self._tm),
                "tout": tout_now,
                "solar": sol_now,
                "u": u,
                "w_est": float(self._w_est),
            }
        else:
            self._tm = self.mpc.update_mass_temperature(tin, self._tm)
        self._step_count += 1
        return [np.array([u], dtype=np.float32)]


class _TwoStatePolicyAdaptor:
    """Bridge between evaluate_diffmpc_policy's calling convention and TwoStateDiffMPCPolicy."""

    def __init__(self, mpc, normalization, indices, *, device, stress_profile=None, **kwargs):
        self._inner = TwoStateDiffMPCPolicy(
            mpc,
            indices,
            device=device,
            stress_profile=stress_profile,
            setpoint_profile=kwargs.get("setpoint_profile", None),
            setpoint_bias_profile=kwargs.get("setpoint_bias_profile", None),
            sigma=0.0,
            observer=kwargs.get("observer", "propagate"),
            kalman_process_var=kwargs.get("kalman_process_var", 0.05),
            kalman_measurement_var=kwargs.get("kalman_measurement_var", 0.25),
            kalman_initial_var=kwargs.get("kalman_initial_var", 4.0),
        )

    def reset(self):
        self._inner.reset()

    def predict(self, obs):
        return self._inner.predict(obs)


def run_online_2r2c_rlmpc_no_occupant(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path = "results/raw/rlmpc",
    summary_dir: str | Path = "results/summaries/rlmpc",
    seed: int = 49,
    training_seed: int = 0,
    reward_function=None,
    reward_function_kwargs=None,
    device: str = "auto",
    sigma: float = 0.10,
    clip_eps: float = 0.2,
    ppo_epochs: int = 1,
    minibatch_size: int = 64,
    lr_ppo: float = 5e-3,
    gamma: float = 0.99,
    max_steps: int | None = None,
    label: str = "simple_v4",
    building_name: str = CONTROL_BUILDING_NAME,
    building_metadata_dict: dict[str, Any] | None = None,
    stress_profile=None,
    setpoint_profile=None,
    setpoint_bias_profile=None,
    matched_mpc_cost: dict[str, float] | None = None,
    tm_penalty_weight: float = 0.0,
    two_state_observer: str = "propagate",
    kalman_process_var: float = 0.05,
    kalman_measurement_var: float = 0.25,
    kalman_initial_var: float = 4.0,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
) -> dict[str, Any]:
    """Online 2R2C RLMPC: PPO fine-tuning of cost weights using a 2R2C DiffMPC."""
    import numpy as np
    import pandas as pd
    import torch

    from src.controllers.rlmpc import twostatediffmpc_param_snapshot

    set_experiment_seed(training_seed)
    mpc, checkpoint, torch_device = load_offline_2r2c_checkpoint(checkpoint_path, device=device)
    mpc.tm_penalty_weight = float(tm_penalty_weight)
    if matched_mpc_cost is not None:
        mpc.use_balanced_mpc_cost(**matched_mpc_cost)
    indices = checkpoint["indices"]

    env_probe = make_february_raw_env(
        seed=seed, reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs, building_name=building_name,
        sim_start=sim_start, sim_end=sim_end,
    )
    obs, _ = env_probe.reset(seed=seed)
    if setpoint_profile is None:
        schedule = (
            env_probe.unwrapped.buildings[0]
            .energy_simulation
            .indoor_dry_bulb_temperature_heating_set_point_without_control
        )
        n_steps = max(1, int(sim_end) - int(sim_start) + 1)
        setpoint_profile = [
            float(schedule[min(int(sim_start) + i, len(schedule) - 1)])
            for i in range(n_steps)
        ]
    env_indices = observation_indices_from_names(list(env_probe.unwrapped.buildings[0].active_observations))
    base_dim = len(env_probe.unwrapped.buildings[0].active_observations)

    baseline_model = fit_january_value_baseline(
        dataset_path=dataset_path, indices=env_indices, base_dim=base_dim, gamma=gamma,
    )
    print("Baseline V_pi0 trained.")

    # --- Collect PPO trajectory ---
    env_train = make_february_raw_env(
        seed=seed, reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs, building_name=building_name,
        sim_start=sim_start, sim_end=sim_end,
    )
    policy_stoch = TwoStateDiffMPCPolicy(
        mpc,
        env_indices,
        device=torch_device,
        stress_profile=stress_profile,
        setpoint_profile=setpoint_profile,
        setpoint_bias_profile=setpoint_bias_profile,
        sigma=sigma,
        observer=two_state_observer,
        kalman_process_var=kalman_process_var,
        kalman_measurement_var=kalman_measurement_var,
        kalman_initial_var=kalman_initial_var,
    )

    obs, _ = env_train.reset(seed=seed)
    policy_stoch.reset()
    trajectory = []
    terminated = truncated = False
    step_count = 0
    while not (terminated or truncated):
        action = policy_stoch.predict(obs)
        obs_next, reward, terminated, truncated, _ = env_train.step(action)
        r = float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward)
        trajectory.append({"obs": obs.copy(), "action": action[0].copy(), "reward": r, "next_obs": obs_next.copy()})
        obs = obs_next
        step_count += 1
        if max_steps is not None and step_count >= int(max_steps):
            break

    advantages, returns = compute_advantages_with_baseline(
        trajectory, baseline_model, indices=env_indices, base_dim=base_dim, gamma=gamma,
    )
    print(f"Collected steps: {len(trajectory)}")
    print(f"Adv mean: {advantages.mean():.6f}  std: {advantages.std():.6f}")

    params_before = twostatediffmpc_param_snapshot(mpc)

    # --- PPO update (cost weights only) ---
    ppo_history: list[dict] = []
    if getattr(mpc, "cost_mode", "tracking") == "balanced_mpc":
        print("PPO update skipped: 2R2C RLMPC inner cost is fixed to the matched MPC objective.")
        ppo_history.append({"ppo_epoch": 0, "loss": float("nan"), "skipped": True})
    else:
        mpc.train()
        opt = torch.optim.Adam(mpc.parameters(), lr=lr_ppo)
        for ep in range(int(ppo_epochs)):
            idx_all = np.arange(len(trajectory))
            np.random.shuffle(idx_all)
            ep_losses = []
            for start in range(0, len(idx_all), int(minibatch_size)):
                batch_idx = idx_all[start : start + int(minibatch_size)]
                opt.zero_grad()
                batch_loss = torch.tensor(0.0, device=torch_device)
                for bi in batch_idx:
                    step = trajectory[bi]
                    o = np.asarray(step["obs"], dtype=np.float32).reshape(-1)
                    tin = float(o[env_indices["tin"]])
                    tm = float(mpc.tm_buffer.item())
                    tout_now = float(o[env_indices["tout"]])
                    sol_now = float(o[env_indices["solar"]])
                    tsp = float(o[env_indices["tsp"]])
                    T = mpc.T
                    tout_p = [o[p] for p in env_indices["tout_p"]]
                    sol_p = [o[p] for p in env_indices["solar_p"]]
                    tout_h = ([tout_now] + tout_p)[:T]
                    sol_h = ([sol_now] + sol_p)[:T]
                    tout_h += [tout_h[-1]] * (T - len(tout_h))
                    sol_h += [sol_h[-1]] * (T - len(sol_h))

                    d_seq = torch.tensor([[tout_h[i], sol_h[i]] for i in range(T)],
                                         dtype=torch.float32, device=torch_device).unsqueeze(0)
                    sp_seq = torch.full((1, T, 1), tsp, dtype=torch.float32, device=torch_device)
                    tin_t = torch.tensor([[tin]], dtype=torch.float32, device=torch_device)
                    tm_t = torch.tensor([[tm]], dtype=torch.float32, device=torch_device)

                    stress_seq = None
                    if stress_profile:
                        gs_val = float(stress_profile[min(bi, len(stress_profile) - 1)])
                        stress_seq = torch.full((1, T), gs_val, dtype=torch.float32, device=torch_device)

                    u_opt, _, _ = mpc(tin_t, tm_t, d_seq, sp_seq, stress_seq)
                    u_old = torch.tensor(float(step["action"][0]), device=torch_device)
                    adv = float(advantages[bi])
                    log_prob_new = -0.5 * ((u_opt - u_old) / max(sigma, 1e-6)).pow(2)
                    log_prob_old = torch.tensor(-0.5 * ((u_old - u_old) / max(sigma, 1e-6)).pow(2), device=torch_device)
                    ratio = torch.exp(log_prob_new - log_prob_old)
                    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    ppo_loss = -torch.min(ratio * adv, clipped * adv)
                    batch_loss = batch_loss + ppo_loss

                batch_loss = batch_loss / max(1, len(batch_idx))
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(mpc.parameters(), 1.0)
                opt.step()
                ep_losses.append(float(batch_loss.item()))
            ppo_history.append({"ppo_epoch": ep, "loss": float(np.mean(ep_losses))})

    mpc.eval()
    params_after = twostatediffmpc_param_snapshot(mpc)

    # Evaluate final deterministic policy
    result = evaluate_diffmpc_policy(
        mpc=mpc,
        normalization=checkpoint["normalization"],
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        device=torch_device,
        max_steps=max_steps,
        building_name=building_name,
        policy_cls=_TwoStatePolicyAdaptor,
        policy_kwargs={
            "stress_profile": stress_profile,
            "setpoint_profile": setpoint_profile,
            "setpoint_bias_profile": setpoint_bias_profile,
            "observer": two_state_observer,
            "kalman_process_var": kalman_process_var,
            "kalman_measurement_var": kalman_measurement_var,
            "kalman_initial_var": kalman_initial_var,
        },
    )

    online_info = {
        "train_reward_total": float(sum(s["reward"] for s in trajectory)),
        "train_steps": int(len(trajectory)),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "q_track_before": float(params_before["q_track"]),
        "r_u_before": float(params_before["r_u"]),
        "q_track_after": float(params_after["q_track"]),
        "r_u_after": float(params_after["r_u"]),
        "model_type": "2r2c",
        "two_state_observer": str(two_state_observer),
        "sp_bias": 0.0,
        "sp_bias_before": 0.0,
        "sp_bias_after": 0.0,
    }
    if str(two_state_observer).strip().lower() == "kalman":
        online_info["kalman_process_var"] = float(kalman_process_var)
        online_info["kalman_measurement_var"] = float(kalman_measurement_var)
        online_info["kalman_initial_var"] = float(kalman_initial_var)
    if building_metadata_dict:
        online_info.update(building_metadata_dict)
    if stress_profile is not None:
        online_info["grid_stress_profile_mean"] = float(np.mean(stress_profile))
    if setpoint_profile is not None:
        online_info["setpoint_profile_length"] = int(len(setpoint_profile))
        online_info["setpoint_profile_mean"] = float(np.mean(setpoint_profile)) if len(setpoint_profile) else 0.0
    if setpoint_bias_profile is not None:
        online_info["setpoint_bias_profile_length"] = int(len(setpoint_bias_profile))
        online_info["setpoint_bias_profile_mean"] = float(np.mean(setpoint_bias_profile)) if len(setpoint_bias_profile) else 0.0
    if matched_mpc_cost is not None:
        online_info.update(
            {
                "mpc_cost_mode": "balanced_mpc",
                "matched_mpc_cost": True,
                "matched_mpc_comfort_weight": float(matched_mpc_cost["comfort_weight"]),
                "matched_mpc_energy_weight": float(matched_mpc_cost["energy_weight"]),
                "matched_mpc_peak_weight": float(matched_mpc_cost["peak_weight"]),
                "matched_mpc_ramp_weight": float(matched_mpc_cost["ramp_weight"]),
                "matched_mpc_shift_weight": float(matched_mpc_cost.get("shift_weight", 0.0)),
                "matched_mpc_comfort_mode": str(matched_mpc_cost.get("comfort_mode", "band")),
                "matched_mpc_comfort_track_weight": float(matched_mpc_cost.get("comfort_track_weight", float("nan"))),
                "matched_mpc_comfort_cold_weight": float(matched_mpc_cost.get("comfort_cold_weight", float("nan"))),
                "matched_mpc_comfort_hot_weight": float(matched_mpc_cost.get("comfort_hot_weight", float("nan"))),
                "matched_mpc_comfort_cold_tolerance": float(
                    matched_mpc_cost.get("comfort_cold_tolerance", float("nan"))
                ),
                "matched_mpc_comfort_hot_tolerance": float(
                    matched_mpc_cost.get("comfort_hot_tolerance", float("nan"))
                ),
                "matched_mpc_comfort_track_scale": float(
                    matched_mpc_cost.get("comfort_track_scale", float("nan"))
                ),
                "matched_mpc_comfort_cold_scale": float(
                    matched_mpc_cost.get("comfort_cold_scale", float("nan"))
                ),
                "matched_mpc_comfort_hot_scale": float(
                    matched_mpc_cost.get("comfort_hot_scale", float("nan"))
                ),
                "matched_mpc_time_bias_guard_band": float(
                    matched_mpc_cost.get("time_bias_guard_band", float("nan"))
                ),
                "matched_mpc_time_bias_guard_penalty": float(
                    matched_mpc_cost.get("time_bias_guard_penalty", float("nan"))
                ),
                "matched_mpc_power_scale": float(matched_mpc_cost["power_scale"]),
                "matched_mpc_power_per_unit_action": float(matched_mpc_cost["power_per_unit_action"]),
                "matched_mpc_normalize_terms": bool(matched_mpc_cost["normalize_terms"]),
            }
        )

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

    summary_path = Path(summary_dir) / "no_occupant" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([metrics])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if "case" in existing.columns and "case" in new_row.columns:
            existing = existing[existing["case"] != metrics["case"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(summary_path, index=False)

    print("February reward after PPO:", metrics["reward_total"])
    print(f"Saved summary to {summary_path}")
    return {"result": result, "metrics": metrics, "ppo_history": ppo_history}
