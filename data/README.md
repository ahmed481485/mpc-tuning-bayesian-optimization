# Cached thesis results

Python pickles written by the experiment scripts. All weights are stored in
log10 scale, and all costs are positive (lower is better).

## `ground_truth/` – true cost on a dense grid ([ground_truth.py](../experiments/ground_truth.py))

| File                      | Grid weights | Objective | Horizon N | Grid      |
|---------------------------|--------------|-----------|-----------|-----------|
| `1d_q3_time.pkl`          | q3           | time      | 50        | 300       |
| `1d_q3_time_N25.pkl`      | q3           | time      | 25        | 200       |
| `1d_q3_time_N100.pkl`     | q3           | time      | 100       | 200       |
| `1d_q1_time.pkl`          | q1           | time      | 50        | 200       |
| `1d_q3_effort.pkl`        | q3           | effort    | 50        | 200       |
| `1d_q3_combined.pkl`      | q3           | combined  | 50        | 200       |
| `2d_q1_q3_time.pkl`       | q1, q3       | time      | 50        | 31 × 31   |
| `2d_q1_q3_effort.pkl`     | q1, q3       | effort    | 50        | 31 × 31   |
| `2d_q3_R_effort.pkl`      | q3, R        | effort    | 50        | 31 × 31   |
| `2d_q3_R_combined.pkl`    | q3, R        | combined  | 50        | 31 × 31   |

Keys: `x_real`, `y_real` (all grid points), `x_real_penalized` / `y_real_penalized`
(unsuccessful swing-ups), `x_real_success` / `y_real_success`. The 1D thesis files
also contain `time_real_taken`, the episode end time.

## `bo_vs_random/` – 5D BO vs. random search ([compare_bo_random.py](../experiments/compare_bo_random.py))

`5d_time.pkl`, `5d_effort.pkl`, `5d_combined.pkl`: 5 runs with 250 evaluations each.
Keys: `all_bo_x_lists`, `all_bo_lists`, `all_random_x_lists`, `all_random_lists`.
Each random run starts from the same 50 initial points as the BO run with the same index.

## `pareto/` – Pareto front ([pareto_front.py](../experiments/pareto_front.py))

`1d_q3.pkl`: 20 values of α × 200 values of q3. Keys: `alphas`, `total_cost`,
`timeToGoal_cost`, `minimizeInput_cost`.
