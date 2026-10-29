# 星地多分支证据拒识器：预登记

run `20261029-phase3-star-consensus-s2026092701-r01`，当前为发布前登记，不是实测成功。完整 Phase3 目标 ACTIVE，按最新 v3 双90%/同输入闭集损失2pp/宏A_K-R_U-H执行；旧实验不追改。

实际代码 commit `652091c76546716d4ef687f771854ef0560dbcb6`。33项相关测试、35subtests PASS；唯一独立P0/P1审查PASS（独立定向6tests/3subtests与主检查有交集，不重复计unique）。一次本地runtime validate及N607 preflight PASS。实际checkpoint无query smoke是source predictor第一步，成功立即进入已登记矩阵。

方法、数据权限、固定矩阵、预算、源晋级和技术停止规则见[最小设计](../../../docs/PAMF_PHASE3_STAR_CONSENSUS_PLAN_20261029.md)。L6300/OE6134训练，knownV27000仅阈值，novel3150仅冻结后的源设备留出评分；同五源RX，不是跨RX目标证据。原F0/完整6身份头冻结，DURA只作为复用缓存祖先；target/clean/新信道生成均0。

唯一主P1/mlp16_oe；2paths×7scores全部预测后独立truth-last。known42组CK≥90%、原F0closed−open≤2pp，unknownoverall/5RX/6scene≥90%；任何控制不能替换主候选。每path linear/MLP各200CPU更新，共800；F/head/G更新0。源科学失败仍全评，不阈值/预算重跑，不直接访问target。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_consensus_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-consensus-s2026092701-r01-release/runtime.json`。
环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-consensus-s2026092701-r01-release`。
run/output：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-consensus-s2026092701-r01`；logs：同root的`logs/`；local：`E:/type10-7/local_artifacts/pamf_phase3_star_consensus_20261029_r01`。
输入cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`，只读已完成四角色；私有novel truth只由独立CPU scorer打开。
GPU：从0..7选实际free≥4096MB一张，单GPU预测child，同卡本任务新增≤2，外国owner保持；实际卡/PID/CWD/argv/GPU/log增长启动后独立记录。preflight实际各卡free为15GB以上，不能把此刻空闲当实际部署峰值。

产物为`prediction/complete.json`和`prediction/by_path/`，紧凑预测在`prediction/predict/*.npz`；独立score在`score/results.json`及`score/complete.json`，完整readerloss与训练元数据在各path readouts.json。只同步小型报告/代码/登记，IQ/模型/原始日志/预测不提交。

REJECTED_EXTRA_GATE：不新增设计审批、重复数据验证、文件hash/receipt、报告审查或报告完整性发布门槛。Git代码commit固定实现，一份release archive做一次local/remote SHA比较和一次compile；用户自主授权内继续。
