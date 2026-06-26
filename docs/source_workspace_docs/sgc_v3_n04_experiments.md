# SGC v3 N04 Experiments

Base checkpoint:

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth
```

Launch script:

```bash
bash code/scripts/run_sgc_v3_n04_gpu0_3.sh --dry-run
bash code/scripts/run_sgc_v3_n04_gpu0_3.sh
```

Available GPUs: `0,1,2,3`.

## Experiment Matrix

| ID | GPU | Config | Mode | Epochs | Purpose |
|---|---:|---|---|---:|---|
| SGCV3-10_blrc_mixed | 0 | `SGC/configs/sgc_v3_blrc_only.yaml` | `blrc_only` | 40 | Safest logit boundary calibration baseline. |
| SGCV3-12_ipfa_mixed | 1 | `SGC/configs/sgc_v3_ipfa_only.yaml` | `ipfa_only` | 40 | Low-rank identity-preserving feature adaptation without logit edits. |
| SGCV3-14_ipfa_blrc_mixed | 2 | `SGC/configs/sgc_v3_safe.yaml` | `ipfa_blrc` | 50 | Main source-only SGC v3 path. |
| SGCV3-20_target_mixed | 3 | `SGC/configs/sgc_v3_target_adapt.yaml` | `target_adapt` | 30 | Pseudo-label target adaptation path. |

All runs use:

```text
dataset=wisig
wisig_domain=rx_day
train_days=0,1
test_days=2,3
train_rxs=0,1,2,3,4,5,6
test_rxs=7,8,9,10,11
batch_size=256
eval_batch_size=256
sat_train_scenario=mixed_orbit
```

## Direct Commands

```bash
cd code

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python -u -m SGC.v3.train_sgc_v3 \
  --teacher_ckpt /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth \
  --config SGC/configs/sgc_v3_blrc_only.yaml \
  --mode blrc_only \
  --epochs 40 \
  --lr_sgc 1e-4 \
  --sat_train_scenario mixed_orbit \
  --output_dir ../runs/sgc_v3_n04/SGCV3-10_blrc_mixed

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" python -u -m SGC.v3.train_sgc_v3 \
  --teacher_ckpt /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth \
  --config SGC/configs/sgc_v3_ipfa_only.yaml \
  --mode ipfa_only \
  --epochs 40 \
  --lr_sgc 1e-4 \
  --sat_train_scenario mixed_orbit \
  --output_dir ../runs/sgc_v3_n04/SGCV3-12_ipfa_mixed

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python -u -m SGC.v3.train_sgc_v3 \
  --teacher_ckpt /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth \
  --config SGC/configs/sgc_v3_safe.yaml \
  --mode ipfa_blrc \
  --epochs 50 \
  --lr_sgc 1e-4 \
  --sat_train_scenario mixed_orbit \
  --output_dir ../runs/sgc_v3_n04/SGCV3-14_ipfa_blrc_mixed

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD" python -u -m SGC.v3.train_sgc_v3 \
  --teacher_ckpt /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth \
  --config SGC/configs/sgc_v3_target_adapt.yaml \
  --mode target_adapt \
  --epochs 30 \
  --lr_sgc 5e-5 \
  --sat_train_scenario mixed_orbit \
  --output_dir ../runs/sgc_v3_n04/SGCV3-20_target_mixed
```
