# WISER-RF P3-Primary稳定快速版预登记与机制核查报告

## 当前结论

- 新run ID：`wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1`。
- 当前最高状态：`LOCAL_VERIFIED`；尚未同步、smoke、启动pilot、打开query或连接truth。
- Git代码提交：`31381958f8075686f5d9410822f5f42428cc417f`；远端分支OID已独立核对一致。
- 本地聚焦验证：112项通过；四个运行模块`py_compile`通过；`git diff --check`通过；唯一一次独立P0/P1审查结论为`READY`。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`、固定`capsule_id/split_id`、support-only训练与选择、全部support状态冻结后只读打开query、prediction完整后由独立scorer连接truth。

## 域训练过久修复

旧run的`p3-smoke`错误复用了正式N6的10000步预算，运行约6小时后才在Stage2发现非有限梯度。新实现做了三层有界化：

1. smoke默认只执行Stage1、三个Stage2分支和Stage3各1步，共5个optimizer step；相对旧smoke上限缩短2000倍。
2. 正式P3配置从`[1500,2000,2500]`改为`[40,60,80]`；N6最坏路径为`40+3×60+80=300`步，相对旧10000步缩短33.3倍。
3. 正式Stage2只有Stage1通过support-only门槛才进入；Stage3只有选中Stage2分支继续通过才进入。每10步诊断，至少20步后连续2次无改善即早停。query不参与步数、门槛、分支或插值选择。

本run的正式arms冻结为`N0,N2,N3,N4,N5,N6`。旧`N1=WISER-A`采用独立8000步旧损失、已退出候选且规则禁止成为冠军；重新运行不会验证新机制，只会恢复主要时延，因此本run不启动N1，也不对N1作同row新结论。

## 失败机制与修复闭环

|问题|旧状态|新实现|本地证据|真实证据缺口|
|---|---|---|---|---|
|D92内层Adam零二阶矩高阶导数|合法单热点几何前向有限、反向15360个NaN|零点安全开平方，保持正数前向值，零点导数有限|先红后绿的高阶梯度测试|真实checkpoint smoke|
|D92 score RMS退化|RMS为0后除零|零/数值退化RMS显式拒绝|退化定点测试|真实动态是否再触发|
|梯度首错不可定位|只检查最终`parameter.grad`|依次检查loss、primary、auxiliary、projected、combined与dual，异常先写JSONL再抛出|正常/失败事件测试|真实训练轨迹|
|`diagnostic_interval`未消费|6小时无中间artifact|正式循环每周期写`training_progress.jsonl`并flush|事件消费测试|远端artifact增长|
|Stage2/3无条件运行|所有分支固定跑满|support-only门槛、耐心早停、未过门槛不深训|门槛/早停测试|各真实场景实际步数|
|选中状态可能未回载|Stage2选择后模型可能停在最后评估分支|最终显式回载选中state并refreeze|runner状态测试|prediction receipt|
|机制诊断不完整|缺按类zero-id、block trace、单模态风险、块级梯度夹角|补齐上述字段及canonical correlations|P3/runner测试|真实数值|
|资源证据缺失|无训练/预测资源闭环|阶段/总训练耗时、峰值RSS/VRAM、状态字节、prediction耗时进入审计/receipt|序列化测试|真实资源值|

没有加入dual cap、梯度裁剪、`nan_to_num`或静默跳步，因为现有证据没有证明对偶放大是首因；这些操作会改变科学方法动力学，不能作为无证据补丁。

## 设计机制/设置/参数全面核查

- `N2`：5-fold cross-fitted old-only精确D92主损失，已实现并由正式入口消费。
- `N3`：每类风险、soft floor、非负对偶更新，已实现并由正式入口消费。
- `N4`：6类目标中心与26×6 int8源域类中心的共享域流形，已实现并由正式入口消费。
- `N5`：P3主梯度与source-head、target-prototype、domain-manifold辅助梯度的全局冲突投影，已实现并由正式入口消费。
- `N6`：identity–FFT互补/冗余与identity能量约束，已实现并由正式入口消费。
- 24个`WISERP3TrainingConfig`字段均在正式runner中有实际消费点；新增早停字段有严格类型/范围校验；旧`20260831`配置保持原样，新配置独立为`configs/wiser_rf_p3_primary_20260901_stablefast.json`。
- source类内低秩协方差摘要仍是设计明确的后续项，本轮未实现；现有int8域×类中心已足够验证共享域流形。
- C/ABC、旧classwise VSW、ASAM、SWA和大权重网格已明确移出主矩阵，不属于“漏启动”。
- Target25、完整125与阶段B没有启动；它们必须等待本pilot科学门槛，不能由技术修复提前授权。

## 冻结输入与资源

- outer：`rx_3_19__seed_713102__k_10__new_5`；receiver=`3-19`；seed=`713102`；K=`10`；new-count=`5`。
- scenes：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- arms：`N0,N2,N3,N4,N5,N6`；共18个prediction/receipt。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`。
- source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`。
- source binding：`configs/wiser_rf_adv3b02_source_binding.json`。
- P3 config：`configs/wiser_rf_p3_primary_20260901_stablefast.json`。
- 物理GPU0；`CUDA_VISIBLE_DEVICES=0`后程序使用`cuda:0`。2026-09-01 01:50CST只读盘点8张GPU均无compute-app，本run加入后不超过用户授权的每卡3个训练任务。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- release：`wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958.tar.gz`。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/pilot.out`。
- score：`<run-root>/score`，必须在prediction完整后由独立进程创建。

## 冻结命令

远端CWD固定为上述release根。

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260901_stablefast.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --source-summary /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/smoke --device cuda:0 --runtime-commit 31381958f8075686f5d9410822f5f42428cc417f --arm N6 --scenario leo_clear_weak --smoke-stage-steps 1 1 1
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260901_stablefast.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --source-summary /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1/pilot --device cuda:0 --runtime-commit 31381958f8075686f5d9410822f5f42428cc417f --arms N0 N2 N3 N4 N5 N6
```

## 停止规则与预期artifact

- 只允许因协议/query越权、错误split/receiver/seed/K/scene、错误checkout/CWD、run root冲突、非有限loss/gradient/dual、确定性重复异常、prediction不完整或scorer连接错误停止。
- 不因低性能、负收益、未晋级或缺少非必要字段停止。
- smoke预期：`smoke/training_progress.jsonl`、`smoke/smoke_result.json`，且`query_opened=false/query_rows_used=0`。
- pilot预期：18个`training_audit.json`、适配state、18个完整`predictions.npz`与receipt、`support_audit.json`、`pilot_result.json`。
- prediction完整后另起独立`p3-score-pilot`连接truth，生成详细score与`score_collection.json`。报告绝对Accuracy/BA/floor/NLL、per-class、P1/P2/P3、适应增量、help/harm、训练/预测资源和三场景同row结果。
- 只有三场景pilot门槛通过才授权Target25；Target25通过后才讨论完整125与阶段B。

## 2026-09-01远端发布与启动回执

- release归档本地/远端唯一SHA256均为`c27b1a136dff7e8912cfcb7936ee5d71724dbe49be3b36f84d2244f7b6b25ad9`；远端四个运行模块一次`py_compile`通过。
- 启动前只读复核：8张GPU compute-app为空；新run、log根均不存在。物理GPU0被冻结且未超过每卡3个训练任务。
- 远端owner PID=`3208551`（PPID1），smoke worker PID=`3208559`；worker CWD精确为`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260901_v1_stablefast1_31381958`，cmdline、config、checkpoint、source summary、output root和runtime commit与预登记一致。
- GPU映射：worker PID`3208559`位于物理GPU0 UUID=`GPU-56adac86-77cd-36c9-8770-dbf002650461`，首次采样显存6700MiB。
- 启动36秒时`smoke/training_progress.jsonl`已有3957字节；最新事件为`stage2_time/step1`，loss、primary、auxiliary、projected、combined gradient及dual均有限，`zero_identity_count=0`，`query_rows_used=0`。日志仍为0字节属于stdout缓冲，不影响JSONL增长证据。
- 当前最高状态更新为`RUNNING`。不重复启动、不热修改、不因中间性能停止；下一次检查在smoke预计完成后进行。

## 2026-09-01 03:00CST小时检查

- 真实ADV3B02 checkpoint无query smoke已`PASS`：5个optimizer step完整覆盖Stage1、三个Stage2分支和Stage3，总训练耗时90.466秒；旧smoke约6小时仍未完成，快速闭环已经由真实远端证据确认。
- smoke峰值CUDA分配6395007488字节，进程峰值RSS2075070464字节；最终`zero_identity_count=0`，identity-only、FFT-only、joint OOF风险均为`VALID`，全程`query_rows_used=0/query_opened=false`。
- 同一owner PID`3208551`已按预登记自动进入`p3-pilot`，当前运行约1小时3分；物理GPU0显存8788MiB。
- pilot支持适配阶段已完成5个`training_audit.json`，当前推进到`leo_clear_weak/N6`，对应进度JSONL为8221字节；尚未打开query，因此prediction/receipt为0/18符合support-first顺序。
- 定点扫描未发现`FAILED_NONFINITE`，未发现技术异常。保持运行，不终止、不重启、不热修改、不因性能停止。
