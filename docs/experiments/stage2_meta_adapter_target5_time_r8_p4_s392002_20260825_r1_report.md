# Time-only Rank-8 Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`480e447f3fda3fe5b89067335092bf47c14a1ce0`；GitHub远端分支OID已独立回读一致。
- Phase1：`phase1_adv3b02_meta_adapter_time_r8_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 候选保持P4 FOMAML+Meta-SGD、正式3步support更新和冻结原型余弦判决；相对已闭合time+fusion rank4只保留time adapter并将rank提高到8，用于隔离time方向且增加其容量，不加入fusion／freq，不改变更新步数或判决头。
- 正式bundle严格回读预算5458／1055125，占0.517285%；10个可训练张量只含原编码器`id/dom_backbone.meta_adapter_time`，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`；receiver=`20-1`；operating point固定为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 配置：`configs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1.json`。相对已闭合time+fusion rank4 Target5计划，5个entries逐项不变，只替换`bundle_id`、`checkpoint_path`和`prototype_path`。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ及原capsule／split，不因候选变化重验数据。
- Phase2仅读取合法target support IQ和support标签、正式bundle、冻结原型；不读取任何source／clean样本、source cache、query真值或query角色。
- query只在3步support反向传播完成并冻结模型后逐样本推理，不更新模型、原型、归一化统计或其他状态。
- 同一冻结判决规则比较`DA0_REG0`与`DA1_REG0`；REG0的新类指标为N/A，不引入D92式分类头。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1/checkout`
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1.out`
- Target工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1.json --output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_p4_s392002_20260825_r1`。
- smoke命令：`python code/scripts/smoke_stage2_meta_adapter_no_query.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_p4_s392002_20260825_r1/smoke_config_no_query.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1_smoke --device cuda`。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_p4_s392002_20260825_r1/matrix_config.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1 --device cuda`。
- expected artifacts：`factory_receipt.json`、`smoke_receipt.json`、15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`，以及矩阵级`matrix_receipt.json`和`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与科学门槛

- 本地计划差异断言通过：共15个row，正式steps=3；相对已闭合time+fusion rank4计划，唯一变化键精确为`bundle_id`、`checkpoint_path`、`prototype_path`，5个entries逐项相等。
- 本候选实现阶段已通过262项Meta-Adapter Phase1／Phase2邻近回归和9个生产入口编译；Stage2 factory／runner／scorer代码没有变化，沿用本次会话已通过的69项聚焦回归和11个生产入口编译证据。
- time-only rank8候选唯一一次独立P0/P1审查结论为P0无、P1无；Stage2复用已验证factory／runner／scorer，不增加重复审查。
- 15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

## N607 release验证

- release归档固定当前Git HEAD=`ea966672e6986751d144cdd96c559b13ad438be4`；本地路径为`E:\type10-7\release_archives\stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1_release.tar.gz`。
- 唯一release归档本地／远端一次SHA256均为`30d420f9bcd9aa040d0b8ecad5850686036864476a0c4968a42966f3ff18f8fe`；远端release已解压到预登记checkout，14个相关生产入口编译通过。
- 发布前独立确认release、工厂、smoke、prediction和stdout目标均不存在；Phase1正式bundle和冻结原型均非空，无同名持久进程；GPU0～7均无计算负载。

## Target工厂与真实checkpoint smoke

- Target工厂一次完成15／15个truth-free row，receipt为`TARGET_INPUTS_COMPLETE`；`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- smoke专用配置在本地从首row配置精确删除`query_path`，并通过与首row其余字段完全相等以及无query／source／clean／truth／role键的断言后同步。
- 真实Phase1 time-only rank8 bundle无query smoke一次通过：`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`、严格checkpoint回读、3次真实反向传播、`trainable_fraction=0.005172846819097263`、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`、`performance_result=null`。

## Target5最终结果

- 唯一正式prediction矩阵正常退出并一次完成15／15行；矩阵级状态为`PREDICTIONS_COMPLETE`，`truth_opened=false`、`source_opened=false`。15行均严格加载正式bundle，先完成3次support反向传播后才打开query；`query_opened_before_adaptation=false`、`query_role_opened=false`、`query_truth_opened=false`、`query_state_update_count=0`，可训练比例均为0.517285%。
- prediction完整后才连接三个既有同row truth sidecar；15个`score.json`和`target5_summary.json`均非空并已下载到`E:\type10-7\local_artifacts\meta_adapter_recovery\target5_time_r8_p4_r1_complete_20260825`独立复算。

|场景|DA0_REG0旧类均值|DA1_REG0旧类均值|DA0_REG0 floor|DA1_REG0 floor|均值变化|floor变化|决策变化数|
|---|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|63.33%|63.33%|35.00%|35.00%|0.00pp|0.00pp|0|
|`leo_low_elev_weak`|63.33%|63.33%|35.00%|35.00%|0.00pp|0.00pp|0|
|`leo_rain_weak`|66.67%|66.67%|45.00%|45.00%|0.00pp|0.00pp|0|

- 适配前后每行最大绝对余弦分数变化范围为0.000167698～0.015051037，但15行均无类别决策变化。聚合`mean_delta_pp=0.0`、`floor_delta_pp=0.0`，未达到+1.0pp／+0.5pp门槛，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`，严格不进入Target25。
- 科学解释：增加time adapter容量仍只能改变分数，无法让support梯度跨越旧类决策边界；结合time+fusion rank4同样零决策变化、fusion rank8出现负迁移，下一候选不再机械扩大rank或叠加位置，而应约束support更新方向／尺度，使少层梯度真正朝目标支持集的判别改善方向移动。
