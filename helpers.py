import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from citylearn.citylearn import CityLearnEnv
import math
import seaborn as sns
import matplotlib.ticker as ticker


def select_buildings(
    schema: dict,
    count: int,
    seed: int,
    buildings_to_exclude: list[str] = None,
) -> list[str]:
    """
    Randomly select buildings from an Annex96 / ResStock CityLearn dataset.
    """

    all_buildings = list(schema["buildings"].keys())

    assert 1 <= count <= len(all_buildings), \
        f"count must be between 1 and {len(all_buildings)}."

    np.random.seed(seed)

    # handle exclusions
    if buildings_to_exclude is None:
        buildings_to_exclude = []
    else:
        buildings_to_exclude = list(buildings_to_exclude)

    # filter buildings
    selectable_buildings = [
        b for b in all_buildings if b not in buildings_to_exclude
    ]

    assert len(selectable_buildings) >= count, \
        "Not enough buildings left after exclusions."

    # random selection
    selected = np.random.choice(
        selectable_buildings, size=count, replace=False
    ).tolist()

    # optional: deterministic ordering for reproducibility
    selected = sorted(selected)

    return selected


def select_simulation_period(
    schema: dict,
    dataset_dir: Path,
    count: int,
    seed: int,
    simulation_periods_to_exclude: list[tuple[int, int]] = None
) -> tuple[int, int]:
    """
    Randomly select simulation start and end time steps
    covering `count` days.
    """

    assert 1 <= count <= 365, "count must be between 1 and 365."

    np.random.seed(seed)

    # pick any building to infer total time steps
    building_name = list(schema["buildings"].keys())[0]
    filename = schema["buildings"][building_name]["energy_simulation"]
    filepath = dataset_dir / filename

    # total available time steps
    time_steps = pd.read_csv(filepath).shape[0]

    # candidate start steps (aligned to day boundaries)
    step_size = 24 * count
    simulation_start_time_step_list = np.arange(0, time_steps - step_size, step_size)

    # exclude periods if provided
    if simulation_periods_to_exclude is not None:
        exclude_starts = [s for s, _ in simulation_periods_to_exclude]
        simulation_start_time_step_list = np.setdiff1d(
            simulation_start_time_step_list,
            exclude_starts
        )

    # randomly select a start step
    # simulation_start_time_step = np.random.choice(simulation_start_time_step_list)
    simulation_start_time_step = 0
    simulation_end_time_step = simulation_start_time_step + step_size - 1

    return simulation_start_time_step, simulation_end_time_step


def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    kpis = env.unwrapped.evaluate().copy()

    # Friendly names (only remap what exists)
    kpi_map = {
        'cost_total': 'Cost',
        'carbon_emissions_total': 'Emissions',
        'all_time_peak_average': 'All-time peak',
        'daily_one_minus_load_factor_average': '1 - load factor',
        'annual_normalized_unserved_energy_total': 'Unserved energy'
    }

    kpis = kpis[kpis['cost_function'].isin(kpi_map.keys())].copy()
    kpis['kpi'] = kpis['cost_function'].map(kpi_map)

    return kpis

def get_all_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """
    Return ALL KPIs from CityLearn evaluate(),
    removing duplicates and NaNs.
    """
    kpis = env.unwrapped.evaluate().copy()

    # Remove NaNs (some KPIs not defined for some setups)
    kpis = kpis[~kpis["value"].isna()].copy()

    # Remove duplicated cost_function/level/name rows
    kpis = kpis.drop_duplicates(
        subset=["cost_function", "level", "name"]
    )

    # Make KPI names prettier
    kpis["kpi"] = (
        kpis["cost_function"]
        .str.replace("_", " ")
        .str.title()
    )

    return kpis


def plot_building_kpis(envs: dict[str, CityLearnEnv]) -> plt.Figure:

    kpis_list = []

    for env_id, env in envs.items():
        kpis = get_all_kpis(env)
        kpis = kpis[kpis["level"] == "building"].copy()
        kpis["env_id"] = env_id
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True)

    kpi_names = sorted(kpis["kpi"].unique())

    column_count = 3
    row_count = math.ceil(len(kpi_names) / column_count)

    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 3 * row_count),
        squeeze=False,
    )

    for ax, kpi in zip(axes.flatten(), kpi_names):

        data = kpis[kpis["kpi"] == kpi]

        sns.barplot(
            x="value",
            y="name",
            hue="env_id",
            data=data,
            ax=ax,
        )

        ax.set_title(kpi)
        ax.set_xlabel("")
        ax.set_ylabel("")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

        ax.legend().set_visible(False)

    axes.flatten()[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        framealpha=0.0,
    )

    plt.tight_layout()
    return fig


def plot_district_kpis(envs: dict[str, CityLearnEnv]) -> plt.Figure:

    kpis_list = []

    for env_id, env in envs.items():
        kpis = get_all_kpis(env)
        kpis = kpis[kpis["level"] == "district"].copy()
        kpis["env_id"] = env_id
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True)

    # Sort KPIs for clean plotting
    kpis = kpis.sort_values("kpi")

    env_count = len(envs)
    kpi_count = len(kpis["kpi"].unique())

    figsize = (8, 0.35 * env_count * kpi_count)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.barplot(
        x="value",
        y="kpi",
        hue="env_id",
        data=kpis,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")

    for s in ["right", "top"]:
        ax.spines[s].set_visible(False)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        framealpha=0.0,
    )

    plt.tight_layout()
    return fig


def plot_building_load_profiles(
    envs: dict[str, CityLearnEnv], daily_average: bool = None
) -> plt.Figure:
    """Plots building-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    daily_average: bool, default: False
        Whether to plot the daily average load profile.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    daily_average = False if daily_average is None else daily_average
    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.unwrapped.buildings[i].net_electricity_consumption
            y = np.reshape(y, (-1, 24)).mean(axis=0) if daily_average else y
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_ylabel('kWh')

        if daily_average:
            ax.set_xlabel('Hour')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        else:
            ax.set_xlabel('Time step')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig


def plot_district_load_profiles(
    envs: dict[str, CityLearnEnv], daily_average: bool = None
) -> plt.Figure:
    """Plots district-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    daily_average: bool, default: False
        Whether to plot the daily average load profile.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    daily_average = False if daily_average is None else daily_average
    figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for k, v in envs.items():
        y = v.unwrapped.net_electricity_consumption
        y = np.reshape(y, (-1, 24)).mean(axis=0) if daily_average else y
        x = range(len(y))
        ax.plot(x, y, label=k)

    ax.set_ylabel('kWh')

    if daily_average:
        ax.set_xlabel('Hour')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    else:
        ax.set_xlabel('Time step')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)

    plt.tight_layout()
    return fig


def plot_battery_soc_profiles(envs: dict[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level battery SoC profiles fro different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = np.array(v.unwrapped.buildings[i].electrical_storage.soc)
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('SoC')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.set_ylim(0.0, 1.0)

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig

def plot_simulation_summary(envs: dict[str, CityLearnEnv]):

    print('#'*8 + ' BUILDING-LEVEL ' + '#'*8)

    print('Building-level KPIs:')
    _ = plot_building_kpis(envs)
    plt.show()

    print('Building-level simulation period load profiles:')
    _ = plot_building_load_profiles(envs)
    plt.show()

    print('Building-level daily-average load profiles:')
    _ = plot_building_load_profiles(envs, daily_average=True)
    plt.show()

    print('Battery SoC profiles:')
    _ = plot_battery_soc_profiles(envs)
    plt.show()

    print('#'*8 + ' DISTRICT-LEVEL ' + '#'*8)

    print('District-level KPIs (ALL):')
    _ = plot_district_kpis(envs)
    plt.show()

    print('District-level simulation period load profiles:')
    _ = plot_district_load_profiles(envs)
    plt.show()

    print('District-level daily-average load profiles:')
    _ = plot_district_load_profiles(envs, daily_average=True)
    plt.show()




def plot_comfort(env: CityLearnEnv, building_idx: int = 0):
    b = env.unwrapped.buildings[building_idx]

    # Indoor temperature
    T_in = np.array(b.indoor_dry_bulb_temperature)

    # Setpoint (heating == cooling in this dataset)
    T_sp = np.array(b.indoor_dry_bulb_temperature_heating_set_point)

    # Fixed comfort band
    T_low = 20.0
    T_high = 24.0

    timesteps = np.arange(len(T_in))

    plt.figure(figsize=(12, 4))

    plt.plot(timesteps, T_in, label='Indoor temperature', linewidth=2)
    plt.plot(timesteps, T_sp, '--', label='Setpoint', linewidth=2)

    plt.fill_between(
        timesteps,
        T_low,
        T_high,
        color='gray',
        alpha=0.2,
        label='Comfort band (20–24 °C)'
    )

    plt.xlabel('Time step')
    plt.ylabel('Temperature [°C]')
    plt.title(b.name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def plot_comfort_with_lstm_1step(env: CityLearnEnv, dyn, building_idx: int = 0):
    b = env.unwrapped.buildings[building_idx]

    # true signals
    T_in = np.array(b.energy_simulation.indoor_dry_bulb_temperature)
    T_sp = np.array(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point)

    timesteps = np.arange(len(T_in))

    # LSTM metadata
    names = dyn.input_observation_names
    mins = np.asarray(dyn.input_normalization_minimum, dtype=np.float32)
    maxs = np.asarray(dyn.input_normalization_maximum, dtype=np.float32)
    lookback = dyn.lookback

    iTin = names.index("average_indoor_air_temperature")

    def norm(x, i):
        return (x - mins[i]) / (maxs[i] - mins[i] + 1e-12)

    def denorm(x, i):
        return x * (maxs[i] - mins[i]) + mins[i]

    # ---- bootstrap lookback buffers with TRUE history ----
    model_input = []
    for i, name in enumerate(names):
        buf = []
        for k in range(lookback + 1):
            t = max(0, k)

            if name == "average_indoor_air_temperature":
                raw = T_in[t]
            elif name == "heating_load":
                raw = b.energy_simulation.heating_demand[t]
            elif name == "outdoor_air_temperature":
                raw = b.weather.outdoor_dry_bulb_temperature[t]
            else:
                raw = 0.0

            buf.append(norm(raw, i))
        model_input.append(buf)

    h = dyn.init_hidden(batch_size=1)

    Tin_pred_1step = np.full_like(T_in, np.nan)

    # ---- one-step-ahead prediction loop ----
    for t in range(lookback, len(T_in) - 1):

        # update buffers with TRUE values at time t
        for i, name in enumerate(names):
            if name == "average_indoor_air_temperature":
                raw = T_in[t]
            elif name == "heating_load":
                raw = b.energy_simulation.heating_demand[t]
            elif name == "outdoor_air_temperature":
                raw = b.weather.outdoor_dry_bulb_temperature[t]
            else:
                raw = 0.0

            model_input[i] = model_input[i][-lookback:] + [norm(raw, i)]

        # build LSTM input (CityLearn convention)
        X = []
        for i, name in enumerate(names):
            if name == "average_indoor_air_temperature":
                X.append(model_input[i][:-1])
            else:
                X.append(model_input[i][1:])
        X = torch.tensor(np.array(X).T, dtype=torch.float32)[None, :, :]

        with torch.no_grad():
            y_norm, h = dyn(X, h)

        Tin_pred_1step[t + 1] = denorm(float(y_norm.item()), iTin)

    # ---- plotting ----
    plt.figure(figsize=(13, 4))

    plt.plot(timesteps, T_in, label="Indoor temperature (EnergyPlus)", linewidth=2)
    plt.plot(timesteps, T_sp, "--", label="Setpoint", linewidth=2)
    plt.plot(
        timesteps,
        Tin_pred_1step,
        # ":",
        label="LSTM 1-step-ahead prediction",
        linewidth=2,
    )

    plt.fill_between(
        timesteps,
        20.0,
        24.0,
        color="gray",
        alpha=0.2,
        label="Comfort band (20–24 °C)",
    )

    plt.xlabel("Time step")
    plt.ylabel("Temperature [°C]")
    plt.title(f"{b.name} — EnergyPlus vs LSTM (1-step)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_env(**kwargs):
    """
    Create a CityLearn environment.

    Keyword arguments can override any CityLearnEnv parameter,
    e.g. reward_function, active_actions, random_seed, etc.
    """

    base_kwargs = dict(
        schema=str(SCHEMA_PATH),
        root_directory=str(DATASET_DIR),
        central_agent=CENTRAL_AGENT,
        buildings=BUILDINGS,
        simulation_start_time_step=SIMULATION_START_TIME_STEP,
        simulation_end_time_step=SIMULATION_END_TIME_STEP,
        # HVAC ONLY — no batteries
        active_actions=[
            'heating_device',
            # or 'cooling_device'
            # or 'cooling_or_heating_device'α
        ],
    )

    # Override defaults with user-provided arguments
    base_kwargs.update(kwargs)

    return CityLearnEnv(**base_kwargs)


def run_env(env, agent):
    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        actions = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(actions)

    return env

