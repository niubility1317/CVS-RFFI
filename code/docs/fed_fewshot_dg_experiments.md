# Few-Shot Federated Domain-Generalization Experiments

This matrix targets the current setting:

```text
federated learning + 10% labeled train split + 90% validation split + day/rx domain generalization
```

Use `--wisig_train_ratio 0.1`. In the existing WiSig split code, validation is the held-out tail from the same train day/receiver pool after the guard gap, so this corresponds to roughly 10% train and 90% validation inside the source domains.

## Main Questions

1. Does federated training help when only 10% source-domain samples are labeled?
2. Is `receiver_day` too fragmented for few-shot FL, or is coarse `receiver` more stable?
3. Does FedProx reduce client drift under small local data?
4. How sensitive is FedProx to `mu` under receiver/day client drift?
5. How far is CE-only FL from the current strong centralized DG recipe?
6. What happens when the strong `BEX02_fishr002_mixed_e170` DG recipe is moved inside each federated client's local objective?
7. Does a receiver-client FedProx objective with GRL receiver-adversarial loss learn stronger receiver-agnostic TX features?
8. Is baseline-style supervised clean+satellite view expansion stronger than CVS feature-consistency satellite training inside CVS-RFFI?

## Experiment Groups

`SMOKE` runs two tiny jobs to validate the launcher and environment.

Recommended execution order:

```text
FED_BASE / CORE -> FED_DG -> CENTRAL
```

This runs pure federated algorithm baselines first, then the federated BEX02 strong-DG variants, and only then the centralized baselines.

`FED_BASE` / `CORE` runs the pure federated algorithm baselines:

- `FSDG12_fedavg_rxday`: main FL baseline.
- `FSDG12A_fedavg_rxday_local3`: FedAvg local-epoch control for the local3 FedProx diagnostic.
- `FSDG13_fedprox_rxday_mu001`: weak FedProx.
- `FSDG14_fedprox_rxday_mu01`: medium FedProx.
- `FSDG14A_fedprox_rxday_mu01_local3`: medium FedProx with stronger local drift.
- `FSDG14B_fedprox_rxday_mu1`: strong FedProx diagnostic point from the common `mu` sweep.
- `FSDG15_fedavg_rx`: coarser receiver clients.
- `FSDG16_fedavg_rxday_frac05`: partial client participation.
- `FSDG17_fedprox_rxday_frac05_mu01`: FedProx under partial participation, testing proximal control and client sampling together.

`FED_DG` ports the strongest centralized BEX02-style recipe into federated local training:

- `FSDG18_fedavg_rxday_bex02dg`: FedAvg with `fl_local_objective=bex02_dg`.
- `FSDG19_fedprox_rxday_bex02dg_mu001`: weak FedProx with the same BEX02 local objective.
- `FSDG1A_fedprox_rxday_bex02dg_mu01`: medium FedProx with the same BEX02 local objective.
- `FSDG1B_fedprox_rxday_bex02dg_mu1`: strong FedProx diagnostic point with the same BEX02 local objective.
- `FSDG49_fedprox_receiver_ra_bex02_cvs_sat`: receiver-client FedProx with `fl_local_objective=receiver_agnostic_bex02`, GRL receiver loss, and CVS satellite consistency.
- `FSDG50_fedprox_receiver_ra_bex02_baseline_sat`: the same receiver-agnostic FedProx objective, but with the baseline supervised clean+satellite view expansion migrated into CVS-RFFI.

Federated jobs now default to 170 communication rounds and 2 local epochs, so the FL training horizon is no longer shorter than the centralized 170-epoch reference. The BEX02/DG jobs use augmentation, MixStyle, `sat_train_scenario=mixed_orbit`, satellite classification weight `lambda_sat_cls=0.10`, satellite consistency weight `lambda_sat_cons=0.00`, `lambda_fishr=0.02`, and `fishr_min_domains=4`. All launcher jobs evaluate star-ground channel robustness through `--eval_sat_channel --eval_sat_on main --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit`.

`CENTRAL` runs the centralized training strategies last as reference baselines:

- `FSDG02_centralized_ce`: CE-only, no augmentation/MixStyle/DG losses.
- `FSDG03_centralized_base_aug`: current backbone recipe with augmentation but no explicit DG losses.
- `FSDG04_centralized_mixstyle`: MixStyle-only DG control.
- `FSDG05_centralized_fishr`: Fishr-only DG control.
- `FSDG06_centralized_sat`: satellite-consistency route without Fishr.
- `FSDG07_centralized_strong_dg`: current strong centralized recipe.

FedProx follows the MLSys 2020 "Federated Optimization in Heterogeneous Networks" baseline: compared with FedAvg, each client keeps a proximal penalty to the current global model during local optimization. In this code path, `--train_mode fedprox --fedprox_mu MU` applies the penalty through `compute_fedprox_loss`, while `--train_mode fedavg` is the `mu=0` reference. The official FedProx repository recommends tuning `mu` for each dataset, with common candidates `{0.001, 0.01, 0.1, 0.5, 1}`: https://github.com/litian96/FedProx. If FedAvg and FedProx look identical, check the `[FED]` line's `prox` and `prox/cls` fields; with very small `mu` or only one local epoch, the proximal penalty can be numerically tiny and FedProx effectively behaves like FedAvg.

`CLIENTS` isolates client granularity.

Random seeds are fixed to `1337`; this matrix no longer runs a seed sweep.

## How To Run

Dry run:

```bash
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan CORE --dry-run
```

Smoke:

```bash
conda activate ssr-gpu
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan SMOKE --gpu-ids 0
```

Pure federated baseline first:

```bash
conda activate ssr-gpu
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FED_BASE --gpu-ids 0,1,2,3,4,5
```

Federated BEX02/DG sweep second:

```bash
conda activate ssr-gpu
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FED_DG --gpu-ids 0,1,2
```

Centralized strategy sweep last:

```bash
conda activate ssr-gpu
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan CENTRAL --gpu-ids 0,1,2
```

Full sweep:

```bash
conda activate ssr-gpu
bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FULL --gpu-ids 0,1,2,3,4,5
```

Override defaults:

```bash
FEWSHOT_RATIO=0.1 FL_ROUNDS=200 FL_LOCAL_EPOCHS=3 \
  bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FED_BASE --gpu-ids 0,1,2,3
```

## What To Compare

Primary metrics:

- `test_unseen_day_unseen_rx` / `strict_udu`
- `test_seen_day_unseen_rx`
- `test_unseen_day_seen_rx`
- validation accuracy, because validation is large in this 10%/90% split

Main comparisons:

```text
FSDG12 FedAvg receiver_day  vs FSDG10 centralized CE
FSDG13/FSDG14 FedProx      vs FSDG12 FedAvg
FSDG15 receiver clients    vs FSDG12 receiver_day clients
FSDG16 partial clients     vs FSDG12 all clients
FSDG17 FedProx partial     vs FSDG16 FedAvg partial
FSDG11 strong centralized  vs all current CE-only FL jobs
FSDG02-FSDG07 centralized  explains which centralized DG ingredient is strongest under 10% data
FSDG18/FSDG19/FSDG1A      test whether BEX02-style local DG plus FedProx closes the gap to FSDG11
FSDG49 vs FSDG50          isolates CVS satellite consistency vs baseline supervised satellite-view augmentation
```

Interpretation note: use `fl_local_objective=ce` for pure FL algorithm controls, `fl_local_objective=bex02_dg` for the strong DG local objective, and `fl_local_objective=receiver_agnostic_bex02` when each receiver is a client and the GRL receiver head should remove receiver information from the TX feature. The active federated methods are FedAvg and FedProx.
