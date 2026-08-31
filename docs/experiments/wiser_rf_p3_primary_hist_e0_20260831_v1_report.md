# WISER-RF P3-Primary最小预登记报告

## 状态与版本

- run ID：`wiser_rf_p3_primary_hist_e0_20260831_v1`。
- 当前状态：`LOCAL_VERIFIED`。本报告不声明已构建release、已连接N607、已启动pilot、已生成prediction或已获得性能结果。
- 运行时代码冻结提交：`e669536eca19cd39154c6232d31f59bc7af0d7e8`（`fix: bind WISER Target25 champion runtime`）。该提交与后续仅包含追踪表和本报告镜像的文档提交分离；文档提交由Task9发布步骤产生并单独核验。
- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。远端环境、具体GPU索引、GPU占用和路径可见性：`PENDING_TASK10_PREFLIGHT`。
- 用户授权的资源上限：每张GPU最多3个训练实验；pilot实际GPU分配必须在Task10只读盘点后冻结，不得在本报告中虚构。

## 冻结矩阵与输入绑定

- arm：`N0,N1,N2,N3,N4,N5,N6`。N0为冻结ADV3B02+精确old-only D92基线；N1为旧WISER A对照；N2至N6依次加入P3-Primary、类别风险/floor、共享域流形、P3主导梯度投影、identity–FFT互补与能量约束。
- pilot outer：`rx_3_19__seed_713102__k_10__new_5`；receiver=`3-19`；seed=`713102`；K=`10`；new-count=`5`。
- scene：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。每个arm必须独立从同一checkpoint fresh-load；共7×3=21个prediction及对应receipt。
- 协议句柄必须匹配`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id`与`split_id`。训练、阶段选择、插值和超参数选择固定`query_rows_used=0`；query仅在support状态完整冻结后只读打开，truth仅由独立scorer在prediction完整后连接。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`。
- source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`。
- source binding：`configs/wiser_rf_adv3b02_source_binding.json`；P3配置：`configs/wiser_rf_p3_primary_20260831.json`。

## 不可覆盖输出与命令

- pilot run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260831_v1/pilot`。
- pilot log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260831_v1`。
- score root：`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260831_v1/score`。
- GPU占位符`cuda:PENDING_TASK10_PREFLIGHT`只能在Task10盘点后替换为实际卡号。

```text
python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <pilot-run-root>/smoke --device cuda:PENDING_TASK10_PREFLIGHT --runtime-commit e669536eca19cd39154c6232d31f59bc7af0d7e8 --arm N6 --scenario leo_clear_weak
python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <pilot-run-root> --device cuda:PENDING_TASK10_PREFLIGHT --runtime-commit e669536eca19cd39154c6232d31f59bc7af0d7e8 --arms N0 N1 N2 N3 N4 N5 N6
python code/scripts/run_stage2_wiser_pilot.py p3-score-pilot --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --prediction-root <pilot-run-root> --output-root <score-root> --runtime-commit e669536eca19cd39154c6232d31f59bc7af0d7e8 --arms N0 N1 N2 N3 N4 N5 N6
```

仅当pilot产生完整21个prediction/receipt、独立评分完成且出现唯一冠军时，才允许条件性Target25：

```text
python code/scripts/run_stage2_wiser_target25.py prepare --source-manifest <manifest> --pilot-marker <score-root>/score_collection.json --output-root <target25-root> --phase target25 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json
python code/scripts/run_stage2_wiser_target25.py run-shard --manifest <target25-root>/manifest.json --shard-index <index> --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --p3-config configs/wiser_rf_p3_primary_20260831.json --output-root <target25-root>/prediction --device cuda:PENDING_TASK10_PREFLIGHT
python code/scripts/run_stage2_wiser_target25.py score-shard --manifest <target25-root>/manifest.json --shard-index <index> --prediction-root <target25-root>/prediction --output-root <target25-root>/score
python code/scripts/run_stage2_wiser_target25.py analyze --manifest <target25-root>/manifest.json --score-root <target25-root>/score --output-root <target25-root>/analysis
```

## 预期artifact、停止规则与科学门槛

- smoke必须证明真实checkpoint可构造N0至N6、`query_opened=false`、`query_rows_used=0`、D92五类同构，并在结束时冻结全部模型参数。smoke通过后立即进入pilot；不创建额外授权artifact。
- pilot的预期artifact是support audit、21个prediction/receipt、独立detailed score、pilot decision marker及资源记录。每个评分row报告实际query数、Accuracy、BA、floor、NLL、per-class、百分点变化、help/harm、P1/P2/P3诊断和训练/预测时延、峰值VRAM/RSS、状态大小。
- 直接技术停止仅限协议或query/truth泄漏、错误split/receiver/seed/K/scene、输出根冲突、可微与精确D92同构失败、非有限loss/gradient、双模态退化、prediction不完整、scorer绑定错误、进程归属不清或确定性重复异常。不得因低性能、负收益或未过科学门槛停止健康任务。
- pilot门槛：候选P3 BA三scene中位提升≥3pp、任一scene不低于-0.5pp、P3 floor中位变化≥0、low-elev floor不下降、zero-id=0、联合协方差条件数不超过冻结基线2倍、至少两个scene正向翻转多于负向翻转，且任一scene的P1/P2下降不超过2pp。
- Target25仅在pilot冠军存在后运行固定5receiver×5seed×3scene=25outer/75scene unit。其门槛为P3 BA中位提升≥3pp、每个scene家族中位变化≥0、10%分位≥-2pp、overall及low-elev floor中位不下降、至少4/5receiver与4/5seed聚合变化为正，并继续满足zero-id、条件数和prediction翻转要求。
- 只有Target25通过才授权三个K10切片`new5/new10/new20`的225scene扩展；只有该K10扩展仍通过，才可进入Stage B。Stage B默认冻结`phi_D`并训练注册专用`phi_R`，后续须报告`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`四状态。

## 本地证据边界

本地完整聚焦套件覆盖11个测试文件，共165项通过，约284秒；`py_compile`覆盖P3、runner、pilot、scoring、Target25和两个CLI；`run_stage2_wiser_pilot.py --help`及`run_stage2_wiser_target25.py --help`均通过；`git diff --check`通过。这些证据只证明`LOCAL_VERIFIED`软件集成，不证明真实checkpoint smoke、N607资源、prediction闭合、truth-last评分或任何性能提升。
