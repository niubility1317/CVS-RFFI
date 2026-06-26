# CVS十二小时项目快照

- 生成时间：2026-06-26T05:24:24Z UTC
- 本地时间戳：`20260626_132424`
- 来源工作区：`E:\type10-7`
- GitHub发布工作区：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 同步文件数：2861
- 跳过文件数：67
- 生成文件清单：`experiment_records/CV-SincNet/snapshot_manifest_latest.json`

## 已同步范围

- 核心代码：`code/`、`baselines/`、`paper_reproduction/`、`tests/`。
- 项目控制文件：`docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。
- 自动化工具：`tools/`中允许公开的`.py`、`.md`、`.ps1`文件。
- 启动脚本：`scripts/launchers/run_*.sh`。
- 最近实验证据：`experiment_records/CV-SincNet/latest/`下的报告、metrics、score table、manifest、matrix、validation和summary类小文件。

## 最近报告目录

- `stage2_spaceborne_h06_oldfuse_repair_20260626_122707`
- `stage2_spaceborne_next128dz_20260623_230632`
- `SATVAL_RA_ABLATION_20260527_103712`
- `cvs_sa27_optimization_central_20260527_204005`
- `paper_reproduction_full_clean_sat_finetune_multiseed_n607_20260626_000000`
- `paper_reproduction_full_clean_sat_n607_20260626_000000`
- `stage2_spaceborne_h06_oldrisk_repair2_20260626_103559`
- `stage2_spaceborne_h06_oldrisk_repair_20260626_102738`
- `stage2_spaceborne_h06_oldqual_repair_20260626_082644`
- `stage2_spaceborne_h06_oldbudget_repair_20260626_062237`
- `stage2_spaceborne_h06_oldconf_repair_20260626_041400`
- `stage2_spaceborne_h06_oldgeom_repair_20260626_022740`

## 边界

- 未上传WiSig/ManySig数据集、模型权重、checkpoint、原始大日志、N607凭据或本地密钥。
- `optimizer_execution_registry.jsonl`只上传tail和fingerprint，避免十二小时提交持续膨胀。
- 指标解释必须绑定同一run或同一candidate row，不能把不同row的单项最大值拼成结论。
- clean view只能作为control/reference；Stage2-A/B不能声明seen-new identity accuracy。
