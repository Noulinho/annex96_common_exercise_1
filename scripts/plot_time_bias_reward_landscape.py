"""Plot CE1 time-bias reward landscapes over indoor temperature and heating.

The stage reward used by the recent SAC/MPC/RLMPC experiments is immediate in
temperature and heating power. This plot makes the local incentive visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.rewards import load_simple_v4_price_profile


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "reward_landscape"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def stage_cost(
    tin: np.ndarray,
    u: np.ndarray,
    *,
    tsp: float,
    bias: float,
    stress: float,
) -> np.ndarray:
    """Current CE1 time-bias guard stage cost with P_heat / 5 approximated by u."""

    biased_tsp = tsp + bias
    tracking = 3.0 * ((tin - biased_tsp) / 2.0) ** 2
    cold_guard = 60.0 * (np.maximum(tsp - 2.0 - tin, 0.0) / 1.5) ** 2
    hot_guard = 15.0 * (np.maximum(tin - tsp - 2.0, 0.0) / 2.0) ** 2
    comfort = tracking + cold_guard + hot_guard
    heat = stress * u
    return 0.5 * comfort + 0.5 * heat


def one_step_tin_1r1c(
    tin: np.ndarray,
    u: np.ndarray,
    *,
    model_path: Path,
    tout: float = 5.0,
    solar: float = 0.0,
) -> np.ndarray:
    """One-step 1R1C proxy for showing how action enters via future Tin."""

    with model_path.open() as f:
        p = json.load(f)
    return (
        float(p["A"]) * tin
        + float(p["Bd0"]) * float(p["mean_indoor_temperature"])
        + float(p["Bd1"]) * tout
        + float(p.get("Bd2", 0.0)) * solar
        + float(p["Bu"]) * u
    )


def context_rows():
    stress_profile = load_simple_v4_price_profile(simulation_start=744, exp_beta=2.0)
    if not stress_profile:
        stress_profile = [1.0] * 24
    hourly_stress = np.asarray(stress_profile[:24], dtype=float)
    return [
        ("preheat low-ish stress\nhour 3, bias +1", 3, +1.0, float(hourly_stress[2])),
        ("morning reduce peak\nhour 7, bias -1", 7, -1.0, float(hourly_stress[6])),
        ("neutral midday\nhour 12, bias 0", 12, 0.0, float(hourly_stress[11])),
        ("evening reduce\nhour 20, bias -1", 20, -1.0, float(hourly_stress[19])),
    ]


def draw_stage_reward_surface():
    tin = np.linspace(15.0, 25.0, 241)
    u = np.linspace(0.0, 1.0, 201)
    tin_grid, u_grid = np.meshgrid(tin, u)
    tsp = 20.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (title, _hour, bias, stress) in zip(axes.flat, context_rows(), strict=True):
        reward = -stage_cost(tin_grid, u_grid, tsp=tsp, bias=bias, stress=stress)
        im = ax.contourf(tin_grid, u_grid, reward, levels=40, cmap="viridis")
        best_u = u[np.argmax(reward, axis=0)]
        ax.plot(tin, best_u, color="white", lw=2.0, label="best immediate u")
        ax.axvline(tsp, color="black", lw=1.0, alpha=0.45)
        ax.axvline(tsp - 2.0, color="black", lw=1.0, ls="--", alpha=0.45)
        ax.axvline(tsp + 2.0, color="black", lw=1.0, ls="--", alpha=0.45)
        ax.set_title(f"{title}\nstress={stress:.2f}")
        ax.set_xlabel("Tin [C]")
        ax.set_ylabel("u or P_heat/5")
        ax.legend(loc="lower left", fontsize=8)
        fig.colorbar(im, ax=ax, label="immediate reward")

    fig.suptitle("Immediate reward landscape: fixed Tin always prefers less heat")
    fig.tight_layout()
    path = FIG_DIR / "time_bias_guard_immediate_reward_surface.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_one_step_proxy_surface():
    tin = np.linspace(15.0, 25.0, 241)
    u = np.linspace(0.0, 1.0, 201)
    tin_grid, u_grid = np.meshgrid(tin, u)
    tsp = 20.0
    model_path = ROOT / "results" / "models" / "rc" / "1state_rc_245723_january.json"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (title, _hour, bias, stress) in zip(axes.flat, context_rows(), strict=True):
        tin_next = one_step_tin_1r1c(tin_grid, u_grid, model_path=model_path)
        reward = -stage_cost(tin_next, u_grid, tsp=tsp, bias=bias, stress=stress)
        im = ax.contourf(tin_grid, u_grid, reward, levels=40, cmap="magma")
        best_u = u[np.argmax(reward, axis=0)]
        ax.plot(tin, best_u, color="white", lw=2.0, label="best 1-step u")
        ax.axvline(tsp, color="black", lw=1.0, alpha=0.45)
        ax.axvline(tsp - 2.0, color="black", lw=1.0, ls="--", alpha=0.45)
        ax.axvline(tsp + 2.0, color="black", lw=1.0, ls="--", alpha=0.45)
        ax.set_title(f"{title}\nstress={stress:.2f}")
        ax.set_xlabel("current Tin [C]")
        ax.set_ylabel("u")
        ax.legend(loc="lower left", fontsize=8)
        fig.colorbar(im, ax=ax, label="1-step proxy reward")

    fig.suptitle("One-step RC proxy: heating only helps through future Tin")
    fig.tight_layout()
    path = FIG_DIR / "time_bias_guard_one_step_proxy_reward_surface_b8_1r1c.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_action_curves():
    u = np.linspace(0.0, 1.0, 201)
    tsp = 20.0
    tins = [16.0, 18.0, 19.0, 20.0, 21.0, 23.0]
    contexts = [
        ("preheat hour 3", +1.0, context_rows()[0][3]),
        ("peak reduce hour 7", -1.0, context_rows()[1][3]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (title, bias, stress) in zip(axes, contexts, strict=True):
        for tin in tins:
            reward = -stage_cost(np.full_like(u, tin), u, tsp=tsp, bias=bias, stress=stress)
            ax.plot(u, reward, label=f"Tin={tin:.0f} C")
        ax.set_title(f"{title}, stress={stress:.2f}, bias={bias:+.0f}")
        ax.set_xlabel("u or P_heat/5")
        ax.set_ylabel("immediate reward")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("At fixed Tin, immediate reward is a downward line in heating")
    fig.tight_layout()
    path = FIG_DIR / "time_bias_guard_reward_vs_action_curves.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    paths = [
        draw_stage_reward_surface(),
        draw_one_step_proxy_surface(),
        draw_action_curves(),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
