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

## 2026-10-29 G10 源端 consensus 拒识实测结果

`20261029-phase3-star-consensus-s2026092701-r01` 两stage完成；主P1/MLP：**NO_PROMOTION**，总目标 **ACTIVE**。

| path/方法 | A_K% | R_U% | H-score% | C_K% | F0损pp | 源gate |
| --- | --- | --- | --- | --- | --- | --- |
| P0/msp | 75.0593 | 4.3810 | 8.2787 | 96.4778 | 0.7519 | False |
| P0/maxlogit | 75.1778 | 4.0635 | 7.7102 | 96.7778 | 0.6333 | False |
| P0/margin | 74.8778 | 6.3175 | 11.6519 | 96.4926 | 0.9333 | False |
| P0/energy | 75.2074 | 3.4603 | 6.6162 | 97.1000 | 0.6037 | False |
| P0/cosine | 75.4222 | 0.0317 | 0.0635 | 99.6111 | 0.3889 | False |
| P0/linear_oe | 75.1630 | 23.0159 | 35.2406 | 98.3481 | 0.6481 | False |
| P0/mlp16_oe | 75.0519 | 25.7778 | 38.3750 | 98.1852 | 0.7593 | False |
| P1/msp | 89.9333 | 15.9683 | 27.1210 | 97.1556 | 1.0481 | False |
| P1/maxlogit | 90.0630 | 14.0635 | 24.3281 | 97.4926 | 0.9185 | False |
| P1/margin | 90.2074 | 7.9048 | 14.5358 | 98.0222 | 0.7741 | False |
| P1/energy | 90.1444 | 10.2540 | 18.4134 | 97.7852 | 0.8370 | False |
| P1/cosine | 90.7519 | 6.5397 | 12.2002 | 99.6481 | 0.2296 | False |
| P1/linear_oe | 90.1593 | 37.1746 | 52.6433 | 97.9556 | 0.8222 | False |
| P1/mlp16_oe | 90.1407 | 44.9206 | 59.9606 | 97.9741 | 0.8407 | False |

v3 known42组CK≥90%/原F0损≤2pp；unknown总体/5sourceRX/6scene均≥90%。原F0/G冻结，2path各linear/MLP200CPU步共800；L6300/OE6134额外membership拟合、V27000仅τ、历史暴露novel3150最后预测/独立truth-last评分，target/U0。
主−P1energy ΔA_K−0.0037pp/ΔR_U＋34.6667pp/ΔH＋41.5472pp，净多拒1092条未知、正确少1条；主−linear ΔR_U＋7.7460pp/ΔH＋7.3173pp，但A_K少5条、macroOSCR低0.1023pp。known42过而unknown12组全失败，仍NO_PROMOTION。
同源RX和既有物理cache复用，不是crossRX target或新TX盲测；Gforward0、F0真实85168行＋synthetic2。只独立捕获driver live，GPU2为pipeline指派，owned_gpu观察为空；预测日志仅终态，无GPU峰值或实时log-growth证据。
完整七方法/42组/逐TX/主−energy与linear配对、800loss、域权重与参数成本见 `docs/PAMF_PHASE3_STAR_CONSENSUS_RESULTS_20261029.md`。AN/HOW=N/A，未进行DA/注册，不凭source单项宣布完整成功。
