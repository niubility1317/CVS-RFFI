# 论文基线从头训练Clean/Satellite完整复现实验报告

## 基本信息

- 实验ID：`paper_reproduction_full_clean_sat_n607_20260626_000000`
- 操作员/agent：Codex
- 目标：不使用CEN51或任何CVS预训练基线模型，只按论文baseline随机初始化从头训练，再用目标域K-shot support做少样本适应/注册，比较clean全链路与satellite全链路两条线。
- 口径：`cvs_extension=true`，结果用于CVS指标对齐；原始论文复现口径仍独立保留。

## 实验矩阵

| 分支 | baseline | channel_line | train | support/query/test | K | query_per_tx | max_steps | GPU |
|---|---|---|---|---|---:|---:|---:|---:|
| `protonet_cda_clean_seed1337` | ProtoNet-CDA | clean_all | clean | clean | 5 | 20 | 200 | 0 |
| `protonet_cda_satellite_seed1337` | ProtoNet-CDA | satellite_all | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak`轮换 | 同三场景逐场景评估 | 5 | 20 | 200 | 1 |
| `feature_separation_clean_seed1337` | Feature Separation | clean_all | clean | clean | 5 | 20 | 200 | 2 |
| `feature_separation_satellite_seed1337` | Feature Separation | satellite_all | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak`轮换 | 同三场景逐场景评估 | 5 | 20 | 200 | 3 |

## CVS数据协议

- `R_s=["1-1","1-19","14-7","18-2","19-2","2-1","2-19"]`
- `R_t=["20-1"]`
- `Y_old=["14-10","14-7","20-15","20-19","6-15","8-20"]`
- `Y_new=["1-16","1-18"]`
- `Y_unknown=["10-1","10-10"]`
- `rho_label=0.1`
- `adaptation_mode="support_prototype_registration"`
- 阈值：`support_only_no_unknown_query`
- 禁止：不使用CEN51权重、不使用unknown query调阈值、不把clean结果写成deployment-primary。

## 本地改动

```text
paper_reproduction/cvs_aligned/protocol.py
paper_reproduction/cvs_aligned/evaluate.py
paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_full_n607.json
paper_reproduction/configs/protonet_cda_cvs_stage2c_satellite_full_n607.json
paper_reproduction/configs/feature_separation_cvs_stage2c_clean_full_n607.json
paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_full_n607.json
run_paper_reproduction_full_clean_sat_n607.sh
tests/test_paper_reproduction_cvs_aligned.py
```

本地快照：
```text
E:\type10-7\code\snapshots\paper_reproduction_full_clean_sat_n607_20260626_000000\
```

关键hash：
```text
17DA52DD0E7C46E39B7BB8C1C087E1D99A8219D5D7760E4CEFCC304BED5D3CFB  protocol.py
E70047BDCE824D3D8951632A31C1BC6BD733ABC10BB939DCBD13C18614830B3B  evaluate.py
FCD459A11A08EE9BEA940724D6DA7316EBF9F460FBD7E89C8BEE25A3D8B8C342  protonet clean config
DEF1E8E37453791FE4F359FCF69E6EED1E52BA03085A8B73E20009EA1D23D9CE  protonet satellite config
73467FD80F43CD27CA209ABFA42EE2C3C97900468EE729073841C093292CD7BC  feature clean config
8ACAE0BDC63035B835C1EC50A5CBC43621DD47247C96323CC77B0D2610FCAAA3  feature satellite config
32BA969D481AF23EE572BF9B36BDB1C26663F34D6AF4325FD76C739CBA4A2E32  launcher
```

## 本地验证

```text
conda activate ssr-gpu; python -m pytest tests/test_paper_reproduction_cvs_aligned.py -q
结果：6 passed；pytest cache目录有Windows权限warning。

conda activate ssr-gpu; python -m py_compile paper_reproduction\cvs_aligned\protocol.py paper_reproduction\cvs_aligned\evaluate.py
结果：PASS

bash -n run_paper_reproduction_full_clean_sat_n607.sh
结果：PASS

四个配置分别执行 --dry-run --formal
结果：全部PASS
```

## 远端计划

同步映射：
```text
paper_reproduction/cvs_aligned/protocol.py -> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/cvs_aligned/protocol.py
paper_reproduction/cvs_aligned/evaluate.py -> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/cvs_aligned/evaluate.py
paper_reproduction/configs/*_full_n607.json -> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/configs/
run_paper_reproduction_full_clean_sat_n607.sh -> /home/szu2070436088/2510044040/CV-SincNet/
```

预期命令：
```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash run_paper_reproduction_full_clean_sat_n607.sh
```

预期输出：
```text
runs/paper_reproduction_full_clean_sat_<timestamp>/launch_manifest.json
runs/paper_reproduction_full_clean_sat_<timestamp>/*/metrics.json
runs/paper_reproduction_full_clean_sat_<timestamp>/*/split_manifest.json
runs/paper_reproduction_full_clean_sat_<timestamp>/*/score_table.csv
logs/paper_reproduction_full_clean_sat_<timestamp>/*.out
```

## 远端同步与验证

N607预检：
```text
时间：2026-06-26 11:00:50 CST
主机：dell-DSS8440
项目根：/home/szu2070436088/2510044040/CV-SincNet
GPU：8 x NVIDIA GeForce RTX 3090；预检时显存约10 MiB/卡
数据：Dataset_WigSig/ManySig.pkl、Dataset_WigSig/ManyTx.pkl存在
```

同步后远端hash与本地记录一致：
```text
17da52dd0e7c46e39b7bb8c1c087e1d99a8219d5d7760e4cefcc304bed5d3cfb  protocol.py
e70047bdce824d3d8951632a31c1bc6bd733abc10bb939dcbd13c18614830b3b  evaluate.py
fcd459a11a08ee9bea940724d6da7316ebf9f460fbd7e89c8bee25a3d8b8c342  protonet clean config
def1e8e37453791fe4f359fcf69e6eed1e52ba03085a8b73e20009ea1d23d9ce  protonet satellite config
73467fd80f43cd27ca209abfa42ee2c3c97900468ee729073841c093292cd7bc  feature clean config
8acae0bdc63035b835c1ec50a5cbc43621dd47247c96323cc77b0d2610fcaaa3  feature satellite config
32ba969d481af23ee572bf9b36bdb1c26663f34d6af4325fd76c739cba4a2e32  launcher
```

远端验证命令：
```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
$PY -m py_compile paper_reproduction/cvs_aligned/protocol.py paper_reproduction/cvs_aligned/evaluate.py
bash -n run_paper_reproduction_full_clean_sat_n607.sh
PYTHONPATH="$PWD:$PWD/code" $PY -m paper_reproduction.cvs_aligned.evaluate --config <four full configs> --dry-run --formal
```

远端验证结果：
```text
py_compile：PASS
bash -n：PASS
四个formal dry-run：PASS
协议检查：Stage2-C有效；R_s/R_t不相交；Y_old/Y_new/Y_unknown互斥；K=5；unknown query不参与阈值；clean为control；satellite/LEO为deployment-primary。
```

## 启动记录

启动命令：
```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash run_paper_reproduction_full_clean_sat_n607.sh
```

启动返回：
```text
RUN_ROOT=runs/paper_reproduction_full_clean_sat_20260626_110328
LOG_ROOT=logs/paper_reproduction_full_clean_sat_20260626_110328
PROTO_CLEAN_PID=3208999
PROTO_SAT_PID=3209002
FEATURE_CLEAN_PID=3209005
FEATURE_SAT_PID=3209008
```

作业映射：

| 分支 | PID | GPU | config | log |
|---|---:|---:|---|---|
| `protonet_cda_clean_seed1337` | 3208999 | 0 | `paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_full_n607.json` | `logs/paper_reproduction_full_clean_sat_20260626_110328/protonet_cda_clean_seed1337.out` |
| `protonet_cda_satellite_seed1337` | 3209002 | 1 | `paper_reproduction/configs/protonet_cda_cvs_stage2c_satellite_full_n607.json` | `logs/paper_reproduction_full_clean_sat_20260626_110328/protonet_cda_satellite_seed1337.out` |
| `feature_separation_clean_seed1337` | 3209005 | 2 | `paper_reproduction/configs/feature_separation_cvs_stage2c_clean_full_n607.json` | `logs/paper_reproduction_full_clean_sat_20260626_110328/feature_separation_clean_seed1337.out` |
| `feature_separation_satellite_seed1337` | 3209008 | 3 | `paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_full_n607.json` | `logs/paper_reproduction_full_clean_sat_20260626_110328/feature_separation_satellite_seed1337.out` |

## 完成状态

远端状态：
```text
四个PID均已退出。
GPU compute app为空。
产物齐全：每个分支均包含metrics.json、resolved_config.json、score_table.csv、split_manifest.json。
错误扫描：未发现Traceback、RuntimeError、ModuleNotFoundError、ValueError、CUDA、OOM或Exception。
```

本地结果快照：
```text
E:\type10-7\automation_reports\CV-SincNet\paper_reproduction_full_clean_sat_n607_20260626_000000\remote_run_snapshot\paper_reproduction_full_clean_sat_20260626_110328\
```

主指标结果：

| run | baseline | line | K | old_acc | seen_new_acc | H_old_new | unknown_FAR | FPR95 | AUROC | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feature_separation_clean_seed1337` | feature_separation_crossrx | clean_all | 5 | 0.408333 | 0.075000 | 0.126724 | 1.000000 | 0.868750 | 0.421719 | fail_unknown_far |
| `feature_separation_satellite_seed1337` | feature_separation_crossrx | satellite_all | 5 | 0.161111 | 0.116667 | 0.130624 | 1.000000 | 0.941667 | 0.383281 | fail_unknown_far |
| `protonet_cda_clean_seed1337` | protonet_cda | clean_all | 5 | 0.791667 | 0.825000 | 0.807990 | 0.900000 | 0.425000 | 0.877656 | fail_unknown_far |
| `protonet_cda_satellite_seed1337` | protonet_cda | satellite_all | 5 | 0.213889 | 0.166667 | 0.186969 | 0.916667 | 0.866667 | 0.587292 | fail_unknown_far |

分场景结果：

| run | scenario | old_acc | seen_new_acc | H_old_new | unknown_FAR | FPR95 | AUROC | old_to_new | new_to_old | unknown_to_old | unknown_to_new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `feature_separation_clean_seed1337` | clean | 0.408333 | 0.075000 | 0.126724 | 1.000000 | 0.868750 | 0.421719 | 0.000000 | 0.925000 | 0.950000 | 0.050000 |
| `feature_separation_satellite_seed1337` | leo_clear_weak | 0.158333 | 0.125000 | 0.139706 | 1.000000 | 0.987500 | 0.372031 | 0.100000 | 0.775000 | 0.875000 | 0.125000 |
| `feature_separation_satellite_seed1337` | leo_low_elev_weak | 0.183333 | 0.075000 | 0.106452 | 1.000000 | 0.950000 | 0.335938 | 0.100000 | 0.825000 | 0.750000 | 0.250000 |
| `feature_separation_satellite_seed1337` | leo_rain_weak | 0.141667 | 0.150000 | 0.145714 | 1.000000 | 0.887500 | 0.441875 | 0.191667 | 0.775000 | 0.875000 | 0.125000 |
| `protonet_cda_clean_seed1337` | clean | 0.791667 | 0.825000 | 0.807990 | 0.900000 | 0.425000 | 0.877656 | 0.008333 | 0.150000 | 0.875000 | 0.025000 |
| `protonet_cda_satellite_seed1337` | leo_clear_weak | 0.225000 | 0.175000 | 0.196875 | 0.925000 | 0.800000 | 0.592187 | 0.233333 | 0.625000 | 0.575000 | 0.350000 |
| `protonet_cda_satellite_seed1337` | leo_low_elev_weak | 0.233333 | 0.200000 | 0.215385 | 0.925000 | 0.893750 | 0.570156 | 0.150000 | 0.725000 | 0.700000 | 0.225000 |
| `protonet_cda_satellite_seed1337` | leo_rain_weak | 0.183333 | 0.125000 | 0.148649 | 0.900000 | 0.906250 | 0.599531 | 0.166667 | 0.675000 | 0.675000 | 0.225000 |

判读：

- 本次实验满足用户要求的两条完整链路：clean全链路与satellite/LEO全链路都覆盖训练、target support、target query/test。
- 本次没有使用CEN51或CVS预训练模型；四个分支均从论文baseline随机初始化训练，随后使用target support做prototype registration/adaptation。
- ProtoNet-CDA在clean control上old/new识别较高，但unknown_FAR=0.900000，拒识失败；不能写成Stage2-C成功。
- ProtoNet-CDA在satellite/LEO上old/new明显下降且unknown_FAR=0.916667，属于星地压力下失败诊断。
- Feature Separation两个分支old/new均较低且unknown_FAR=1.000000，说明该论文baseline实现按当前support-only阈值不能满足CVS未知拒识约束。
- 因所有分支`unknown_FAR>0.05`，最终verdict均为`fail_unknown_far`；这是真实完整复现实验结果，不是部署成功证据。

## 风险和判读边界

- 这是论文baseline从头训练+CVSStage2-C指标评估，不是CEN51/CVS模型复用。
- clean线是control，不是deployment-primary。
- satellite线才是星地压力主线，但仍是WiSig物理启发扰动，不是真实在轨IQ验证。
- 若`unknown_FAR`仍高于0.05，只能作为失败诊断，不能写部署成功。
