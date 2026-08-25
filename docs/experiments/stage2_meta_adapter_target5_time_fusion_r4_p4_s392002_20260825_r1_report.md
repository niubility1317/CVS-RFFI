# Time+Fusion Rank-4 Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`4ad94a9be05e360f4753a479553932c320b1e235`；GitHub远端分支OID已独立回读一致。
- Phase1：`phase1_adv3b02_meta_adapter_time_fusion_r4_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 候选保持P4 FOMAML+Meta-SGD、rank4、正式3步support更新和冻结原型余弦判决；相对fusion-only rank4只增加time adapter位置，不加入freq、不改变更新步数或判决头。
- 正式bundle严格回读预算5780／1055449，占0.547634%；20个可训练张量只含原编码器`id/dom_backbone.meta_adapter_time`与`id/dom_backbone.meta_adapter_fusion`，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`；receiver=`20-1`；operating point固定为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 配置：`configs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1.json`。相对已闭合fusion rank8 Target5计划，5个entries逐项不变，只替换`bundle_id`、`checkpoint_path`和`prototype_path`。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ及原capsule／split，不因候选变化重验数据。
- Phase2仅读取合法target support IQ和support标签、正式bundle、冻结原型；不读取任何source／clean样本、source cache、query真值或query角色。
- query只在3步support反向传播完成并冻结模型后逐样本推理，不更新模型、原型、归一化统计或其他状态。
- 同一冻结判决规则比较`DA0_REG0`与`DA1_REG0`；REG0的新类指标为N/A，不引入D92式分类头。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1/checkout`
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1.out`
- Target工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1.json --output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1`。
- smoke命令：`python code/scripts/smoke_stage2_meta_adapter_no_query.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1/smoke_config_no_query.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_smoke --device cuda`。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1/matrix_config.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1 --device cuda`。
- expected artifacts：`factory_receipt.json`、`smoke_receipt.json`、15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`，以及矩阵级`matrix_receipt.json`和`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与科学门槛

- 本地计划差异断言通过：共15个row，正式steps=3；相对已闭合fusion rank8计划，唯一变化键精确为`bundle_id`、`checkpoint_path`、`prototype_path`，5个entries逐项相等。
- 69项Stage2真实适配／工厂／runner／matrix／handoff／scorer聚焦回归通过，11个生产入口编译通过。测试环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，CWD为本分支worktree根目录。
- time+fusion候选唯一一次独立P0/P1审查结论为P0无、P1无；Stage2复用已验证factory／runner／scorer，不增加重复审查。
- 15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

## N607发布、工厂与真实checkpoint smoke

- release固定提交：`3979e975e3ac3317f10f2fa7f481037f8046e330`；归档：`E:\type10-7\release_archives\stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_release.tar.gz`。
- 唯一release归档本地／远端一次SHA256均为`3a238aeabf155155eb165a7fe6bc63c240b433e0c1fa1e1b755ae179b8caef02`；11个Stage2生产入口在远端`CVS-RFFI`环境编译通过。
- 发布前确认release、工厂、smoke、prediction和stdout目标均不存在；Phase1 bundle／冻结原型非空，无同名持久进程，GPU0～7无计算进程。
- Target工厂一次完成15／15个truth-free row，receipt为`TARGET_INPUTS_COMPLETE`，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- smoke专用配置在本地从首row配置精确删除`query_path`，且不含query／source／clean键；真实bundle无query smoke一次通过：`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`、严格checkpoint回读、3次反向传播、`trainable_fraction=0.005476342296027567`、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`、`performance_result=null`。

## Target5最终结果

- 唯一正式prediction矩阵约10秒完成；外层包装命令末尾因PowerShell转义破坏远端退出码变量而返回非零，但矩阵消费者证据独立回读为`PREDICTIONS_COMPLETE`，故没有重复启动。15／15行均有非空`DA0_REG0`、`DA1_REG0`prediction和receipt，矩阵级`truth_opened=false`、`source_opened=false`。
- 15／15行均严格加载正式bundle，先完成3次support反向传播后才打开query；`query_opened_before_adaptation=false`、`query_role_opened=false`、`query_truth_opened=false`、`query_state_update_count=0`、`source_opened=false`，可训练比例均为0.547634%。
- prediction完整后才连接三个既有同row truth sidecar；15个`score.json`和`target5_summary.json`均非空，无scorer错误。

|场景|DA0_REG0旧类均值|DA1_REG0旧类均值|DA0_REG0 floor|DA1_REG0 floor|均值变化|floor变化|决策变化数|
|---|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|63.33%|63.33%|40.00%|40.00%|0.00pp|0.00pp|0|
|`leo_low_elev_weak`|60.83%|60.83%|30.00%|30.00%|0.00pp|0.00pp|0|
|`leo_rain_weak`|63.33%|63.33%|40.00%|40.00%|0.00pp|0.00pp|0|

- 适配前后每行最大绝对余弦分数变化范围为0.003379732～0.030114323，但15行均没有类别决策变化。聚合`mean_delta_pp=0.0`、`floor_delta_pp=0.0`，未达到+1.0pp／+0.5pp门槛，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`，严格不进入Target25。
- 科学解释：Phase1三类LEO weak的floor增益没有转化为同row目标域决策收益；rank4 time+fusion能改变分数，但更新幅度／方向不足以跨越旧类判决边界。结合fusion rank8在low-elev产生负迁移，下一候选应隔离time方向并提高其容量，而不是继续放大fusion或恢复已失败的tri-site组合。
