# 论文基线Support微调Clean/Satellite多seed完整复现实验报告

## 基本信息

- 实验ID：`paper_reproduction_full_clean_sat_finetune_multiseed_n607_20260626_000000`
- 操作员/agent：Codex
- 目标：修复监督子agent指出的缺口，不使用CEN51或CVS预训练模型；两个论文baseline均从随机初始化开始训练，再用target receiver domain的K-shot support做support-only梯度微调适应；同时跑clean control和satellite/LEO deployment-primary两条完整链路，并补齐多seed均值/标准差。
- 种子：`1337,2027,3407`
- 口径：`cvs_extension=true`，结果用于CVS指标对齐；clean只作control，satellite/LEO才作为deployment-primary压力主线。

## 修正点

| 缺口 | 修正 |
|---|---|
| 少样本适应只有prototype registration | 新增`adaptation_mode="support_head_finetune"`，只用target support训练轻量分类头，不使用unknown query。 |
| ProtoNet CVS评估固定cosine | 新增`prototype_metric`参数；ProtoNet source training仍使用论文原始Euclidean prototypical NLL。 |
| 缺少多seed方差/标准差 | launcher按`1337,2027,3407`三波执行，最终汇总mean/std。 |
| clean与satellite边界需更清晰 | clean配置保持`is_clean_control=true`；satellite配置保持`is_deployment_primary=true`。 |

## 实验矩阵

| baseline | line | train_channel_view | target/support/query view | K | query_per_tx | max_steps | support_finetune_steps |
|---|---|---|---|---:|---:|---:|---:|
| ProtoNet-CDA | clean_all | clean | clean | 5 | 20 | 200 | 100 |
| ProtoNet-CDA | satellite_all | satellite/LEO | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` | 5 | 20 | 200 | 100 |
| Feature Separation | clean_all | clean | clean | 5 | 20 | 200 | 100 |
| Feature Separation | satellite_all | satellite/LEO | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` | 5 | 20 | 200 | 100 |

## CVS数据协议

- `R_s=["1-1","1-19","14-7","18-2","19-2","2-1","2-19"]`
- `R_t=["20-1"]`
- `Y_old=["14-10","14-7","20-15","20-19","6-15","8-20"]`
- `Y_new=["1-16","1-18"]`
- `Y_unknown=["10-1","10-10"]`
- `rho_label=0.1`
- `threshold_scope="support_only_no_unknown_query"`
- 禁止：不使用CEN51权重、不使用unknown query调阈值、不把clean结果写成deployment-primary。

## 本地改动

```text
paper_reproduction/cvs_aligned/evaluate.py
paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_finetune_n607.json
paper_reproduction/configs/protonet_cda_cvs_stage2c_satellite_finetune_n607.json
paper_reproduction/configs/feature_separation_cvs_stage2c_clean_finetune_n607.json
paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_finetune_n607.json
run_paper_reproduction_full_clean_sat_finetune_multiseed_n607.sh
tests/test_paper_reproduction_cvs_aligned.py
```

本地快照：
```text
E:\type10-7\code\snapshots\paper_reproduction_full_clean_sat_finetune_multiseed_n607_20260626_000000\
```

关键hash：
```text
43E77E75F2F6745B82ED89A214061BE180E99597698427DF353EB9E50B5504F6  evaluate.py
2CD365B5E3F34B90EE98B1F610086C3872E5B3AF61FA6025906705F2885BA533  protonet clean finetune config
2FB04B08DACB65DF34BE0FAB1259F839754FD4FA062A6E9246E64304CF65FF30  protonet satellite finetune config
159EAEB61DFE4C5CDF9E0D9A51DC4A01A6FCFF5DBA450A8DB15C93AFAF395B01  feature clean finetune config
C1057E2BA2293ED625316EEF6BAEFCCFC55FBA7E38D0C546CF4E586FC7576A20  feature satellite finetune config
C48C54C8D5860C2B90E42823133CCA9EB9114CFEC10B19939BF404F843BE1E66  launcher
C87F58D0750FCAF03A9E99EC804C72F3CDEAD623D35B3AB7ADEA98C370E95525  tests
```

## 本地验证

```text
conda activate ssr-gpu; python -m pytest tests/test_paper_reproduction_cvs_aligned.py -q
结果：8 passed；pytest cache目录有Windows权限warning。

conda activate ssr-gpu; python -m py_compile paper_reproduction\cvs_aligned\evaluate.py
结果：PASS

bash -n run_paper_reproduction_full_clean_sat_finetune_multiseed_n607.sh
结果：PASS

四个finetune配置分别执行 --dry-run --formal
结果：全部PASS；输出包含adaptation_mode=support_head_finetune、prototype_metric=euclidean、unknown_query_used_for_threshold=false。
```

## N607同步与远端验证

N607预检：
```text
时间：2026-06-26 11:14:57 CST
主机：dell-DSS8440
项目根：/home/szu2070436088/2510044040/CV-SincNet
GPU：8 x NVIDIA GeForce RTX 3090；预检时显存约10 MiB/卡
启动前GPU compute app为空。
```

同步映射：
```text
paper_reproduction/cvs_aligned/evaluate.py -> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/cvs_aligned/evaluate.py
paper_reproduction/configs/*_finetune_n607.json -> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/configs/
run_paper_reproduction_full_clean_sat_finetune_multiseed_n607.sh -> /home/szu2070436088/2510044040/CV-SincNet/
```

远端hash与本地一致：
```text
43e77e75f2f6745b82ed89a214061be180e99597698427df353eb9e50b5504f6  evaluate.py
2cd365b5e3f34b90ee98b1f610086c3872e5b3af61fa6025906705f2885ba533  protonet clean finetune config
2fb04b08dacb65df34be0fab1259f839754fd4fa062a6e9246e64304cf65ff30  protonet satellite finetune config
159eaeb61dfe4c5cdf9e0d9a51dc4a01a6fcff5dba450a8db15c93afaf395b01  feature clean finetune config
c1057e2ba2293ed625316eef6baefccfc55fba7e38d0c546cf4e586fc7576a20  feature satellite finetune config
aa8bd988df6320beda0e29692f5473107c5d9dc57111a92b4c48c7a9a26a3698  launcher
```

远端验证：
```text
py_compile：PASS
bash -n：PASS
四个formal dry-run：PASS
协议检查：Stage2-C有效；R_s/R_t不相交；Y_old/Y_new/Y_unknown互斥；K=5；adaptation_mode=support_head_finetune；unknown query不参与阈值；clean为control；satellite/LEO为deployment-primary。
```

## 启动记录

首次启动命令：
```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash run_paper_reproduction_full_clean_sat_finetune_multiseed_n607.sh
```

首次启动返回：
```text
WAVE_BEGIN seed=1337
SEED=1337 PROTO_CLEAN_PID=3216040 PROTO_SAT_PID=3216043 FEATURE_CLEAN_PID=3216046 FEATURE_SAT_PID=3216049
wait失败：后台PID不是当前shell child。
```

处理：
```text
根因：launcher用p1=$(launch_one ...)命令替换调用函数，函数在subshell中启动后台Python，导致当前shell不能wait这些PID。
措施：未杀运行中的seed1337作业；等待其自然完成。修复launcher为同一shell内设置LAST_PID，并增加SEEDS逗号续跑入口。
修复后launcher hash：c48c54c8d5860c2b90e42823133cca9eb9114cfec10b19939bf404f843be1e66
```

续跑命令：
```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ROOT=runs/paper_reproduction_full_clean_sat_finetune_20260626_111605 \
LOG_ROOT=logs/paper_reproduction_full_clean_sat_finetune_20260626_111605 \
SEEDS=2027,3407 \
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
bash run_paper_reproduction_full_clean_sat_finetune_multiseed_n607.sh
```

续跑返回：
```text
WAVE_BEGIN seed=2027
SEED=2027 PROTO_CLEAN_PID=3217539 PROTO_SAT_PID=3217541 FEATURE_CLEAN_PID=3217543 FEATURE_SAT_PID=3217545
WAVE_END seed=2027
WAVE_BEGIN seed=3407
SEED=3407 PROTO_CLEAN_PID=3217828 PROTO_SAT_PID=3217830 FEATURE_CLEAN_PID=3217832 FEATURE_SAT_PID=3217834
WAVE_END seed=3407
RUN_ROOT=runs/paper_reproduction_full_clean_sat_finetune_20260626_111605
LOG_ROOT=logs/paper_reproduction_full_clean_sat_finetune_20260626_111605
```

完成状态：
```text
metrics.json数量：12
GPU compute app：空
错误扫描：未发现Traceback、RuntimeError、ModuleNotFoundError、ValueError、CUDA、OOM或Exception。
```

本地结果快照：
```text
E:\type10-7\automation_reports\CV-SincNet\paper_reproduction_full_clean_sat_finetune_multiseed_n607_20260626_000000\remote_run_snapshot\paper_reproduction_full_clean_sat_finetune_20260626_111605\
```

## 多seed结果

逐run结果：

| run | seed | baseline | line | old_acc | seen_new_acc | H_old_new | unknown_FAR | FPR95 | AUROC | verdict |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| `feature_separation_clean_seed1337` | 1337 | feature_separation_crossrx | clean_all | 0.183333 | 0.050000 | 0.078571 | 1.000000 | 1.000000 | 0.306563 | fail_unknown_far |
| `feature_separation_clean_seed2027` | 2027 | feature_separation_crossrx | clean_all | 0.316667 | 0.575000 | 0.408411 | 1.000000 | 0.800000 | 0.480781 | fail_unknown_far |
| `feature_separation_clean_seed3407` | 3407 | feature_separation_crossrx | clean_all | 0.241667 | 0.475000 | 0.320349 | 1.000000 | 1.000000 | 0.301719 | fail_unknown_far |
| `feature_separation_satellite_seed1337` | 1337 | feature_separation_crossrx | satellite_all | 0.180556 | 0.100000 | 0.109019 | 1.000000 | 0.866667 | 0.401354 | fail_unknown_far |
| `feature_separation_satellite_seed2027` | 2027 | feature_separation_crossrx | satellite_all | 0.183333 | 0.158333 | 0.124618 | 1.000000 | 0.943750 | 0.459427 | fail_unknown_far |
| `feature_separation_satellite_seed3407` | 3407 | feature_separation_crossrx | satellite_all | 0.155556 | 0.025000 | 0.032000 | 0.983333 | 0.908333 | 0.393854 | fail_unknown_far |
| `protonet_cda_clean_seed1337` | 1337 | protonet_cda | clean_all | 0.816667 | 0.925000 | 0.867464 | 0.625000 | 0.850000 | 0.680312 | fail_unknown_far |
| `protonet_cda_clean_seed2027` | 2027 | protonet_cda | clean_all | 0.883333 | 0.875000 | 0.879147 | 0.725000 | 0.818750 | 0.639375 | fail_unknown_far |
| `protonet_cda_clean_seed3407` | 3407 | protonet_cda | clean_all | 0.883333 | 0.925000 | 0.903687 | 1.000000 | 0.993750 | 0.436719 | fail_unknown_far |
| `protonet_cda_satellite_seed1337` | 1337 | protonet_cda | satellite_all | 0.227778 | 0.191667 | 0.204212 | 0.908333 | 0.950000 | 0.475052 | fail_unknown_far |
| `protonet_cda_satellite_seed2027` | 2027 | protonet_cda | satellite_all | 0.241667 | 0.133333 | 0.166556 | 0.933333 | 0.933333 | 0.451094 | fail_unknown_far |
| `protonet_cda_satellite_seed3407` | 3407 | protonet_cda | satellite_all | 0.216667 | 0.166667 | 0.187473 | 0.958333 | 0.954167 | 0.478229 | fail_unknown_far |

三seed聚合：

| group | baseline | line | seeds | old_acc mean+/-std | seen_new_acc mean+/-std | H_old_new mean+/-std | unknown_FAR mean+/-std | FPR95 mean+/-std | AUROC mean+/-std | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feature_separation_clean` | feature_separation_crossrx | clean_all | 3 | 0.247222+/-0.066840 | 0.366667+/-0.278762 | 0.269110+/-0.170785 | 1.000000+/-0.000000 | 0.933333+/-0.115470 | 0.363021+/-0.102012 | fail_unknown_far |
| `feature_separation_satellite` | feature_separation_crossrx | satellite_all | 3 | 0.173148+/-0.015299 | 0.094444+/-0.066840 | 0.088545+/-0.049587 | 0.994444+/-0.009623 | 0.906250+/-0.038584 | 0.418212+/-0.035890 | fail_unknown_far |
| `protonet_cda_clean` | protonet_cda | clean_all | 3 | 0.861111+/-0.038490 | 0.908333+/-0.028868 | 0.883433+/-0.018488 | 0.783333+/-0.194186 | 0.887500+/-0.093332 | 0.585469+/-0.130437 | fail_unknown_far |
| `protonet_cda_satellite` | protonet_cda | satellite_all | 3 | 0.228704+/-0.012526 | 0.163889+/-0.029266 | 0.186080+/-0.018867 | 0.933333+/-0.025000 | 0.945833+/-0.011024 | 0.468125+/-0.014835 | fail_unknown_far |

分场景三seed聚合：

| group | scenario | seeds | old_acc mean+/-std | seen_new_acc mean+/-std | H_old_new mean+/-std | unknown_FAR mean+/-std | FPR95 mean+/-std | AUROC mean+/-std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `feature_separation_clean` | clean | 3 | 0.247222+/-0.066840 | 0.366667+/-0.278762 | 0.269110+/-0.170785 | 1.000000+/-0.000000 | 0.933333+/-0.115470 | 0.363021+/-0.102012 |
| `feature_separation_satellite` | leo_clear_weak | 3 | 0.177778+/-0.026788 | 0.108333+/-0.101036 | 0.110998+/-0.096672 | 0.983333+/-0.028868 | 0.895833+/-0.084394 | 0.396198+/-0.029528 |
| `feature_separation_satellite` | leo_low_elev_weak | 3 | 0.155556+/-0.061426 | 0.116667+/-0.118145 | 0.097388+/-0.053095 | 1.000000+/-0.000000 | 0.960417+/-0.023662 | 0.386927+/-0.047102 |
| `feature_separation_satellite` | leo_rain_weak | 3 | 0.186111+/-0.089106 | 0.058333+/-0.080364 | 0.057251+/-0.063661 | 1.000000+/-0.000000 | 0.862500+/-0.032476 | 0.471510+/-0.086408 |
| `protonet_cda_clean` | clean | 3 | 0.861111+/-0.038490 | 0.908333+/-0.028868 | 0.883433+/-0.018488 | 0.783333+/-0.194186 | 0.887500+/-0.093332 | 0.585469+/-0.130437 |
| `protonet_cda_satellite` | leo_clear_weak | 3 | 0.230556+/-0.004811 | 0.191667+/-0.038188 | 0.208159+/-0.025331 | 0.941667+/-0.014434 | 0.954167+/-0.023662 | 0.489583+/-0.051077 |
| `protonet_cda_satellite` | leo_low_elev_weak | 3 | 0.205556+/-0.012729 | 0.150000+/-0.075000 | 0.167662+/-0.056773 | 0.908333+/-0.080364 | 0.995833+/-0.007217 | 0.451771+/-0.039424 |
| `protonet_cda_satellite` | leo_rain_weak | 3 | 0.250000+/-0.044096 | 0.150000+/-0.043301 | 0.182420+/-0.023184 | 0.950000+/-0.050000 | 0.887500+/-0.038017 | 0.463021+/-0.034652 |

## 最终判读

- 修正版已补齐监督子agent指出的关键实验缺口：从随机初始化训练；target support-only梯度微调；clean和satellite/LEO两条完整链路；三seed均值/标准差；CVSStage2-C指标完整。
- ProtoNet-CDA在clean control上old/new识别较强，三seed`H_old_new=0.883433+/-0.018488`，但`unknown_FAR=0.783333+/-0.194186`，拒识失败。
- ProtoNet-CDA在satellite/LEO上显著退化，三seed`H_old_new=0.186080+/-0.018867`且`unknown_FAR=0.933333+/-0.025000`，不能作为deployment-primary成功。
- Feature Separation在clean和satellite/LEO两线均未达到可用拒识，`unknown_FAR`接近或等于1.0。
- 因所有分支`unknown_FAR>0.05`，最终verdict均为`fail_unknown_far`。这是真实完整链路复现实验与失败诊断，不是部署成功证据。
