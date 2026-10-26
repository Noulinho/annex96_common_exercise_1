# Research inventory

The full development history is preserved here. For reported thesis results, see [the reading notebooks](../analysis/README.md). Paths are relative to this directory.

## Original analysis notebooks

- [02_mpc_results.ipynb](analysis/02_mpc_results.ipynb): MPC Results.
- [04_rlmpc_results.ipynb](analysis/04_rlmpc_results.ipynb): RLMPC / Gnu-RL Results.
- [08_sac_results.ipynb](analysis/08_sac_results.ipynb): MERLIN-Style SAC Results.
- [10_peak_load_profiles.ipynb](analysis/10_peak_load_profiles.ipynb): Peak Load Profiles.
- [11_no_occupant_single_building_comparison.ipynb](analysis/11_no_occupant_single_building_comparison.ipynb): No-Occupant Single-Building CE1 Target Comparison.
- [12_simple_v4_comparison.ipynb](analysis/12_simple_v4_comparison.ipynb): simple_v4 — Three-Controller Comparison.

## Scratch and deployment notebooks

- [CE1_quickstart.ipynb](notebooks/CE1_quickstart.ipynb) (formerly `notebooks/CE1_quickstart.ipynb`).
- [MPC.ipynb](notebooks/MPC.ipynb) (formerly `notebooks/MPC.ipynb`).
- [citylearn_rlmpc_gnurllike_sac.ipynb](notebooks/citylearn_rlmpc_gnurllike_sac.ipynb) (formerly `notebooks/citylearn_rlmpc_gnurllike_sac.ipynb`).
- [deployment_strategy.ipynb](notebooks/deployment_strategy.ipynb) (formerly `notebooks/deployment_strategy.ipynb`).
- [deployment_strategy_my_run.ipynb](notebooks/deployment_strategy_my_run.ipynb) (formerly `notebooks/deployment_strategy_my_run.ipynb`).
- [my_mpc.ipynb](notebooks/my_mpc.ipynb) (formerly `notebooks/my_mpc.ipynb`).
- [occ.ipynb](notebooks/occ.ipynb) (formerly `notebooks/occ.ipynb`).

## Experiment scripts

Descriptions below come from each script’s original module documentation. Some refer to intermediate results as “final”; use the final run identifier in the main README to distinguish publication results.

- [append_2r2c_diffmpc_action_parity_gate.py](scripts/append_2r2c_diffmpc_action_parity_gate.py): Append 2R2C DiffMPC-vs-explicit-MPC action-parity diagnostics.
- [append_2r2c_fair_visual_result_section.py](scripts/append_2r2c_fair_visual_result_section.py): Replace the DiffMPC parity append with result-style comfort/load plots.
- [append_2r2c_kalman_bad_building_analysis.py](scripts/append_2r2c_kalman_bad_building_analysis.py): Append fixed 2R2C+Kalman diagnostics for the bad-building March MPC runs.
- [append_b17_sac_three_ablation.py](scripts/append_b17_sac_three_ablation.py): Append Building 17 SAC ablation results to the SAC notebook.
- [append_b17_sac_vanilla_norm_check.py](scripts/append_b17_sac_vanilla_norm_check.py): Append Building 17 SAC no-normalization sanity check to SAC notebook.
- [append_bad_building_idrobust_rc_analysis.py](scripts/append_bad_building_idrobust_rc_analysis.py): Append bad-building robust-ID RC comparison for MPC and GNU-RL/RLMPC.
- [append_bad_building_janfeb_rc_holdout_analysis.py](scripts/append_bad_building_janfeb_rc_holdout_analysis.py): Append Jan+Feb PI RC-fit and March holdout diagnostics for bad buildings.
- [append_building11_2r2c_and_march_mpc_comfort.py](scripts/append_building11_2r2c_and_march_mpc_comfort.py): Append Building 11 2R2C fit diagnostics and clean March MPC comfort plots.
- [append_ce1_2r2c_h24_heating_g0_replicate_analysis.py](scripts/append_ce1_2r2c_h24_heating_g0_replicate_analysis.py): Append CE1 2R2C h24 heating-reward GNU-RL replicate comparison.
- [append_ce1_asym_hot_heavy_shift_v1_multi_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v1_multi_analysis.py): Append CE1 asym hot-heavy + shift/preheat v1 results (all 3 controllers) to analysis notebooks.
- [append_ce1_asym_hot_heavy_shift_v2_mpc_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v2_mpc_analysis.py): Append CE1 asym hot-heavy + shift/preheat v2 MPC-only results to experiments/analysis/02_mpc_results.ipynb.
- [append_ce1_asym_hot_heavy_shift_v2_multi_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v2_multi_analysis.py): Append CE1 asym hot-heavy + shift/preheat v2 results (all 3 controllers) to analysis notebooks.
- [append_ce1_asym_hot_heavy_shift_v3_2r2c_mean_target_mpc_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v3_2r2c_mean_target_mpc_analysis.py): Append CE1 v3 2R2C mean-baseline-target MPC results to experiments/analysis/02_mpc_results.ipynb.
- [append_ce1_asym_hot_heavy_shift_v3_2r2c_mpc_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v3_2r2c_mpc_analysis.py): Append CE1 asym hot-heavy + shift/preheat v3 (2R2C) MPC results to experiments/analysis/02_mpc_results.ipynb.
- [append_ce1_asym_hot_heavy_shift_v3_mpc_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v3_mpc_analysis.py): Append CE1 asym hot-heavy + shift/preheat v3 MPC-only results to experiments/analysis/02_mpc_results.ipynb.
- [append_ce1_asym_hot_heavy_shift_v4_mpc_analysis.py](scripts/append_ce1_asym_hot_heavy_shift_v4_mpc_analysis.py): Append CE1 asym hot-heavy + shift/preheat v4 MPC-only results to experiments/analysis/02_mpc_results.ipynb.
- [append_ce1_asym_hot_heavy_v1_analysis.py](scripts/append_ce1_asym_hot_heavy_v1_analysis.py): Append asymmetric hot-heavy MPC diagnostics to the MPC notebook.
- [append_ce1_asym_hot_heavy_v1_multi_analysis.py](scripts/append_ce1_asym_hot_heavy_v1_multi_analysis.py): Append CE1 asym hot-heavy v1 results (all 3 controllers) to analysis notebooks.
- [append_ce1_asym_preheat_peakshift_analysis.py](scripts/append_ce1_asym_preheat_peakshift_analysis.py): Append CE1 asymmetric comfort + peak/shift results to analysis notebooks.
- [append_ce1_comfort_only_sac_analysis.py](scripts/append_ce1_comfort_only_sac_analysis.py): Append Experiment C (comfort-only SAC) results to 08_sac_results.ipynb.
- [append_ce1_daily_windows_analysis.py](scripts/append_ce1_daily_windows_analysis.py): Append no-occupant daily-window load-shaping results to analysis notebooks.
- [append_ce1_district_target_controller_comparison.py](scripts/append_ce1_district_target_controller_comparison.py): Append controller comparison for the CE1 district-target objective.
- [append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis.py](scripts/append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis.py): Append fair CE1 2R2C+Kalman MPC/RLMPC/SAC comparison.
- [append_ce1_heating_4mo_sac_analysis.py](scripts/append_ce1_heating_4mo_sac_analysis.py): Append 4-month heating season SAC comparison to 08_sac_results.ipynb.
- [append_ce1_initial_mean_daily_profiles.py](scripts/append_ce1_initial_mean_daily_profiles.py): Add mean-daily CE1 district profile plots to the early CE1 notebook sections.
- [append_ce1_lagrangian_v1_analysis.py](scripts/append_ce1_lagrangian_v1_analysis.py): Append ce1_lagrangian_v1 SAC (Lagrangian-constrained district reward) results to 08_sac_results.ipynb.
- [append_ce1_mpc_barrier_district_v1_analysis.py](scripts/append_ce1_mpc_barrier_district_v1_analysis.py): Append MPC-only barrier-comfort district-target diagnostics to the MPC notebook.
- [append_ce1_mpc_barrier_district_v2_analysis.py](scripts/append_ce1_mpc_barrier_district_v2_analysis.py): Append MPC-only barrier-comfort district-target v2 diagnostics to the MPC notebook.
- [append_ce1_mpc_barrier_district_v3_analysis.py](scripts/append_ce1_mpc_barrier_district_v3_analysis.py): Append MPC-only barrier-comfort district-target v3 diagnostics to the MPC notebook.
- [append_ce1_mpc_budget_payback_analysis.py](scripts/append_ce1_mpc_budget_payback_analysis.py): Append MPC-only budget/payback load-shaping diagnostics to the MPC notebook.
- [append_ce1_mpc_centered_daily_load_price_analysis.py](scripts/append_ce1_mpc_centered_daily_load_price_analysis.py): Append MPC-only district-target tracking diagnostics to the MPC notebook.
- [append_ce1_mpc_daily_load_price_analysis.py](scripts/append_ce1_mpc_daily_load_price_analysis.py): Append MPC-only district-target tracking diagnostics to the MPC notebook.
- [append_ce1_mpc_district_target_analysis.py](scripts/append_ce1_mpc_district_target_analysis.py): Append MPC-only district-target tracking diagnostics to the MPC notebook.
- [append_ce1_mpc_exp_daily_load_price_analysis.py](scripts/append_ce1_mpc_exp_daily_load_price_analysis.py): Append MPC-only district-target tracking diagnostics to the MPC notebook.
- [append_ce1_mpc_weighted_district_target_analysis.py](scripts/append_ce1_mpc_weighted_district_target_analysis.py): Append MPC-only district-target tracking diagnostics to the MPC notebook.
- [append_ce1_nstep24_sac_analysis.py](scripts/append_ce1_nstep24_sac_analysis.py): Append n-step-24 SAC results to 08_sac_results.ipynb.
- [append_ce1_obsminmax_c3_peak25_shift30_analysis.py](scripts/append_ce1_obsminmax_c3_peak25_shift30_analysis.py): Append CE1 obs-minmax peak/shift results to analysis notebooks.
- [append_ce1_occupant_D010702_analysis.py](scripts/append_ce1_occupant_D010702_analysis.py): Append final occupant-in-loop D010702 controller comparison to notebooks.
- [append_ce1_occupant_p2s2_all_occ_analysis.py](scripts/append_ce1_occupant_p2s2_all_occ_analysis.py): Append final p2 all-occupants controller comparison to notebooks.
- [append_ce1_occupant_sac_occprob_analysis.py](scripts/append_ce1_occupant_sac_occprob_analysis.py): Append SAC occupant-probability reward comparison to 08_sac_results only.
- [append_ce1_phiT50_ramp20_preheat20_analysis.py](scripts/append_ce1_phiT50_ramp20_preheat20_analysis.py): Append CE1 normalized-weight comfort/ramp/preheat results.
- [append_ce1_preheat_credit_analysis.py](scripts/append_ce1_preheat_credit_analysis.py): Append CE1 comfort-corridor + explicit preheat-credit results.
- [append_ce1_rc_rollout_sac_analysis.py](scripts/append_ce1_rc_rollout_sac_analysis.py): Append 1R1C RC-rollout SAC results to 08_sac_results.ipynb.
- [append_ce1_robust_2r2c_kalman_multi_analysis.py](scripts/append_ce1_robust_2r2c_kalman_multi_analysis.py): Append robust 2R2C+Kalman multi-controller comparison to notebooks.
- [append_ce1_sac_b14_jannov_dec_generalization.py](scripts/append_ce1_sac_b14_jannov_dec_generalization.py): Append Building 14 Jan-Nov SAC generalization diagnostic to 08_sac_results.
- [append_ce1_sac_bad_resume20_analysis.py](scripts/append_ce1_sac_bad_resume20_analysis.py): Append SAC bad-building episode-20 continuation diagnostics to notebooks.
- [append_ce1_sac_janfeb20_mar_analysis.py](scripts/append_ce1_sac_janfeb20_mar_analysis.py): Append SAC Jan+Feb training / March deployment analysis to 08_sac_results.
- [append_ce1_sac_noreset_deterministic_january_plots.py](scripts/append_ce1_sac_noreset_deterministic_january_plots.py): Append deterministic January SAC evaluation comfort plots.
- [append_ce1_sac_noreset_jan20_feb_analysis.py](scripts/append_ce1_sac_noreset_jan20_feb_analysis.py): Append CE1 no-reset SAC January-training / February-deployment analysis.
- [append_ce1_sac_noreset_january_episode19_plots.py](scripts/append_ce1_sac_noreset_january_episode19_plots.py): Append final-January SAC training comfort plots for the no-reset run.
- [append_ce1_sac_peakonly_budget_guard_analysis.py](scripts/append_ce1_sac_peakonly_budget_guard_analysis.py): Append SAC peak-only district + heat-budget guard results.
- [append_ce1_sac_replaytail_feb10_mar_analysis.py](scripts/append_ce1_sac_replaytail_feb10_mar_analysis.py): Append SAC replay-tail February continuation / March comparison to 08_sac_results.
- [append_ce1_safe_pi_mpc_cost_analysis.py](scripts/append_ce1_safe_pi_mpc_cost_analysis.py): Append safe-PI + exact-MPC-cost SAC diagnostics to analysis notebooks.
- [append_ce1_safepi_reset_sac_analysis.py](scripts/append_ce1_safepi_reset_sac_analysis.py): Append safe-PI reset SAC results to SAC analysis notebooks.
- [append_ce1_setpoint_energy_1r1c_analysis.py](scripts/append_ce1_setpoint_energy_1r1c_analysis.py): Append corrected setpoint + flat heating-energy MPC analysis cells.
- [append_ce1_setpoint_norm_shift20_analysis.py](scripts/append_ce1_setpoint_norm_shift20_analysis.py): Append CE1 setpoint-normalized load-shifting results to analysis notebooks.
- [append_ce1_supplementary_kpi_cost_audit.py](scripts/append_ce1_supplementary_kpi_cost_audit.py): Historical experiment utility.
- [append_ce1_tdyn_smoke_b06_feb.py](scripts/append_ce1_tdyn_smoke_b06_feb.py): Append the one-building B-spline Tdyn smoke test to 08_sac_results.
- [append_ce1_thesis_clean_load_figures.py](scripts/append_ce1_thesis_clean_load_figures.py): Append thesis-clean load-shape figures for the final CE1 experiments.
- [append_ce1_thesis_final_results.py](scripts/append_ce1_thesis_final_results.py): Generate thesis-ready CE1 result figures/tables from saved artifacts.
- [append_ce1_time_bias_guard_2r2c_kalman_analysis.py](scripts/append_ce1_time_bias_guard_2r2c_kalman_analysis.py): Append CE1 time-biased setpoint guard 2R2C+Kalman comparison results.
- [append_ce1_time_bias_guard_analysis.py](scripts/append_ce1_time_bias_guard_analysis.py): Append CE1 time-biased setpoint guard MPC analysis cells.
- [append_ce1_time_bias_guard_multi_analysis.py](scripts/append_ce1_time_bias_guard_multi_analysis.py): Append time-biased setpoint guard RLMPC/SAC comparison results.
- [append_current_best_mpc_oldrlmpc_sac_comparison.py](scripts/append_current_best_mpc_oldrlmpc_sac_comparison.py): Append current-best MPC, replicated old RLMPC, and latest SAC comparison.
- [append_final_no_occ_normD_controller_comparison.py](scripts/append_final_no_occ_normD_controller_comparison.py): Append final no-occupant normalized-D MPC/RLMPC/SAC comparison.
- [append_final_no_occ_normD_dynamics_rlmpc_comparison.py](scripts/append_final_no_occ_normD_dynamics_rlmpc_comparison.py): Append final no-occupant normalized-D GNU-RL dynamics-adaptation comparison.
- [append_final_no_occ_normD_fixed_adaptive_rlmpc_comparison.py](scripts/append_final_no_occ_normD_fixed_adaptive_rlmpc_comparison.py): Append Feb+March final no-occupant fixed/adaptive GNU-RL comparison.
- [append_mpc_gridstress_ablation_analysis.py](scripts/append_mpc_gridstress_ablation_analysis.py): Compare 2R2C+Kalman MPC with and without the GridStress heat term.
- [append_mpc_normalizedD_no_gridstress_analysis.py](scripts/append_mpc_normalizedD_no_gridstress_analysis.py): Append MPC normalized comfort-only D comparison.
- [append_mpc_rerun_verify.py](scripts/append_mpc_rerun_verify.py): Historical experiment utility.
- [append_old_rlmpc_replication_check.py](scripts/append_old_rlmpc_replication_check.py): Append the old RLMPC exact replication check to analysis notebooks.
- [append_sac_febtrain_march_analysis.py](scripts/append_sac_febtrain_march_analysis.py): Append SAC February-continuation / March-deployment analysis.
- [append_tdyn_gate_sanity_to_sac.py](scripts/append_tdyn_gate_sanity_to_sac.py): Append Tdyn occupancy-gate sanity plots to the SAC results notebook only.
- [compare_old_rlmpc_replicate.py](scripts/compare_old_rlmpc_replicate.py): Compare the original old RLMPC run against a fresh replicate label.
- [create_2r2c_h24_checkpoints.py](scripts/create_2r2c_h24_checkpoints.py): Clone 2R2C DiffMPC checkpoints with a horizon-24 model config.
- [insert_v2_daily_profile_cell.py](scripts/insert_v2_daily_profile_cell.py): Insert missing mean daily district profile figure into the v2 MPC section of 02_mpc_results.ipynb.
- [mark_superseded_fair3_notebook_sections.py](scripts/mark_superseded_fair3_notebook_sections.py): Historical experiment utility.
- [plot_time_bias_reward_landscape.py](scripts/plot_time_bias_reward_landscape.py): Plot CE1 time-bias reward landscapes over indoor temperature and heating.
- [run_2r2c_diffmpc_action_parity.py](scripts/run_2r2c_diffmpc_action_parity.py): Compare first-step 2R2C DiffMPC actions against explicit MPC rollouts.
- [run_ce1_occupant_D010702_full.py](scripts/run_ce1_occupant_D010702_full.py): Run final CE1 occupant-in-loop D010702 controller experiments.
- [run_ce1_occupant_D010702_rlmpc_dyn_mar_continuation.py](scripts/run_ce1_occupant_D010702_rlmpc_dyn_mar_continuation.py): Run CE1 occupant RLMPC adaptive-dynamics March continuation.
- [run_ce1_occupant_sac_occprob.py](scripts/run_ce1_occupant_sac_occprob.py): Run SAC occupant-probability reward ablation for CE1 occupant experiments.
- [run_ce1_sac_janfeb20_mar.py](scripts/run_ce1_sac_janfeb20_mar.py): Train SAC on January+February, then deploy deterministically on March.
- [run_ce1_sac_noreset_jan20_feb.py](scripts/run_ce1_sac_noreset_jan20_feb.py): Run CE1 SAC no-reset sanity experiments for January training -> February deployment.
- [run_ce1_sac_noreset_replaytail_feb10_mar.py](scripts/run_ce1_sac_noreset_replaytail_feb10_mar.py): Resume no-reset SAC with a small January replay tail, train February, deploy March.
- [run_ce1_sac_p3_noreset_jan20_feb.py](scripts/run_ce1_sac_p3_noreset_jan20_feb.py): Run CE1 SAC p3 guard no-reset experiments for January training -> February deployment.
- [run_ce1_sac_p3_noreset_replaytail_feb10_mar.py](scripts/run_ce1_sac_p3_noreset_replaytail_feb10_mar.py): Resume p3 no-reset SAC with a small January replay tail, train February, deploy March.
- [run_sac_deploy_december.py](scripts/run_sac_deploy_december.py): Deploy a trained SAC agent on December without retraining.
- [smoke_tune_safe_pi_one_building.py](scripts/smoke_tune_safe_pi_one_building.py): Small Safe-PI Kp/Ki smoke test for one building.
