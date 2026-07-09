# Annex 96 CE1 Heat-Pump Control Benchmark

This repository contains a deployment-style CityLearn benchmark for heat-pump
control under dynamic, occupant-centric comfort constraints. The work uses the
Vermont portfolio from IEA EBC Annex 96 Common Exercise 1: 25 heating-dominated
single-family homes with hourly simulation data.

The controlled actuator is the space-heating heat pump. All controllers are
evaluated under the same cost formulation, combining scheduled-reference
tracking, asymmetric comfort guards, and time-dependent reference biasing for
load shifting.

## Controllers

The benchmark compares four controller families:

| Controller | Description |
|---|---|
| Safe-PI/RBC | Rule-based PI controller with safety limits, used as a baseline and as SAC replay initialization. |
| SAC | Soft Actor-Critic with one Safe-PI replay episode, January reward training, and February continuation before March deployment. |
| MPC | Model predictive control using a fitted 2R2C thermal model with Kalman-style latent-state correction. |
| GNU-RL | Learning-augmented differentiable MPC with a 1R1C model, imitation initialization, and optional online PPO-style dynamics adaptation. |

## Experiments

Only two final experiments are reported.

| Experiment | Controllers | Evaluation period | Purpose | Status |
|---|---|---|---|---|
| `no_occupant` | Safe-PI/RBC, SAC, MPC, GNU-RL | February and March | Test comfort preservation and aggregate load shaping against the scheduled setpoint. | complete |
| `occupant_present` | Safe-PI/RBC, SAC, MPC, GNU-RL | February and March | Test stochastic occupant feedback, thermostat overrides, B-spline occupancy prediction, and preference-anchored dynamic references. | complete |

The deployment timeline is the same in both experiments. January is treated as a
limited commissioning period. Safe-PI collects stable trajectories, MPC fits its
2R2C model, GNU-RL is initialized through imitation learning, and SAC uses the
first Safe-PI episode as off-policy replay data before reward-based learning.
Controllers are deployed in February. February data are then used for SAC
continuation training and adaptive GNU-RL updates before March deployment.

## Occupant-Centric Setup

The occupant experiments are simulation-based. Stochastic occupant models
generate thermostat override feedback with occupant-specific preferred
temperatures and override sensitivities. These fitted occupant models are not
field-validated human-subject models; they are controlled simulation components
for comparing occupant-dependent controller behavior.

The dynamic-reference mechanism combines:

- fitted stochastic occupant override models,
- B-spline occupancy prediction,
- a preference-anchored dynamic setpoint `Tdyn`,
- occupancy-gated tracking, where the controller tracks the schedule when the
  building is predicted empty and tracks `Tdyn` when occupancy is predicted.

## KPI Summary

The evaluation is building-level first. Load-shape KPIs are computed per
building and normalized against the corresponding baseline building before
aggregation. Comfort violation rates, reference-tracking Kelvin-hours, and
thermostat override starts are also computed per building before averaging.

Headline findings from the final thesis:

- In the no-occupant case, MPC is the most reliable comfort-preserving
  controller. It reaches total discomfort of 1.10% in February and 1.17% in
  March, compared with 41.71% and 34.18% for Safe-PI/RBC.
- GNU-RL follows MPC closely in behavior and comfort, with total discomfort of
  3.94% in February and 3.75% in March. Its online PPO-style adaptation is
  usually mild.
- SAC initially learns a low-heating solution under limited January data,
  producing 26.69% cold discomfort in February. After February continuation
  training, March discomfort falls to 5.70%.
- MPC gives the best non-baseline one-minus load-factor score in both
  no-occupant months and the smoothest non-baseline March ramping score.
- Across representative buildings, MPC generalizes best. GNU-RL is usually
  second, SAC needs continuation data, and Safe-PI/RBC is not robust from a
  comfort perspective.
- In occupant-present simulations, the B-spline dynamic reference reduces
  thermostat override pressure. Average override reduction is about 78% for MPC
  and 72% for GNU-RL, compared with about 14% for Safe-PI/RBC.
- Occupant type matters. Under MPC, tolerant occupants average about 6.1
  overrides per building-month, cold-preferring occupants 16.9, hot-preferring
  occupants 30.4, and sensitive occupants 37.5.

Overall, the results show that useful load shifting cannot be achieved by simply
relaxing thermal comfort. The strongest controllers are those that either
predict the building response well, as MPC does with a 2R2C model and Kalman
correction, or receive enough deployment-like data to learn the response.

## Repository Structure

- `configs/`: building, controller, and scenario definitions.
- `src/`: reusable environment, model, occupant, controller, evaluation, and plotting code.
- `scripts/`: command-line entry points for fitting, running, and summarizing experiments.
- `analysis/`: notebooks and analysis artifacts for result inspection.
- `notebooks/`: exploratory and tutorial notebooks.
- `archive/`: old scratch notebooks and exploratory artifacts.
- `results/`: generated rollouts, summaries, and fitted models; ignored by git.

The thesis and Overleaf export folders are local writing artifacts and are not
part of the GitHub repository.

## Installation

Clone this repository and install the project requirements:

```bash
pip install -r requirements.txt
```

Do not install a separate upstream `citylearn` package for these experiments.
This repository uses the local CityLearn copy and project-specific wrappers.

## Notes

The work is a controlled benchmark for occupant-centric heat-pump control. A real
deployment would require online preference estimation, uncertainty handling, and
validation with real thermostat feedback.
