# Fusion-only Rank-8 Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- Phase1结果提交：`c75fa6f35eafd1df735da73854843611bd8b8b2b`
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 候选保持P4 FOMAML+Meta-SGD、fusion-only层位、正式3步support更新和冻结原型余弦判决，只把上一科学失败候选的bottleneck rank从4提高到8。
- 正式bundle：`phase1_adv3b02_meta_adapter_fusion_r8_p4_s392002_20260825_r1:P4`；严格回读预算5458／1055125，占0.517285%。可训练集合仅含原编码器`id_backbone.meta_adapter_fusion`和`dom_backbone.meta_adapter_fusion`，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`；receiver=`20-1`；operating point固定为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 本次配置：`configs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1.json`。相对已闭合rank-4 Target5计划，5个entries逐项不变，只替换`bundle_id`、`checkpoint_path`和`prototype_path`。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ及原capsule／split，不因候选变化重验数据。
- Phase2仅读取合法target support IQ和support标签、正式bundle、冻结原型；不读取任何source／clean样本、source cache、query真值或query角色。
- query只在3步support反向传播完成并冻结模型后逐样本推理，不更新模型、原型、归一化统计或其他状态。
- 同一冻结判决规则比较`DA0_REG0`与`DA1_REG0`；REG0的新类指标为N/A，不引入D92式分类头。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1/checkout`
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1.out`
- Target工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1.json --output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1`。
- smoke命令：`python code/scripts/smoke_stage2_meta_adapter_no_query.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1/smoke_config_no_query.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1_smoke --device cuda`。`smoke_config_no_query.json`只从首row配置删除`query_path`，先在本地生成再同步。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1/matrix_config.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1 --device cuda`。
- expected artifacts：`factory_receipt.json`、`smoke_receipt.json`、15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`，以及矩阵级`matrix_receipt.json`和`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与科学门槛

- 本地计划差异断言通过：5个entries与rank-4 Target5逐项相等，唯一变更键严格为`bundle_id`、`checkpoint_path`、`prototype_path`；共15个row，正式steps=3。
- 69项Stage2工厂／runner／matrix／handoff／scorer／真实适配聚焦回归通过；11个相关生产入口本地编译通过。测试环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，CWD为本分支worktree根目录。
- rank-8 fusion候选已完成一次独立P0/P1审查及针对唯一CLI默认值P1的一次定点复审，无残留P0/P1；Stage2沿用已验证runner／factory／scorer，不增加重复审查。
- 15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

## N607发布、工厂与真实checkpoint smoke

- release固定提交：`ab047683fac98672634883bd1515a97088827201`；归档：`E:\type10-7\release_archives\stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1_release.tar.gz`。
- release归档本地与远端SHA256均为`b78a6c7f4a399d2c8c4cdb2efd734b723790927b3f30058b1654907f76b19b4a`；11个相关生产入口在远端`CVS-RFFI`环境编译通过。
- 发布前独立确认release、Target工厂、smoke、prediction和stdout目标均不存在；无同名持久进程，GPU0～GPU7无计算进程，项目盘剩余7.3TiB。
- Target工厂一次完成15／15个truth-free row，receipt为`TARGET_INPUTS_COMPLETE`，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- smoke专用配置在本地从首row配置只删除`query_path`，确认不含任何query／source／clean键后同步；未改变正式15-row矩阵。
- 冻结rank-8正式bundle的真实checkpoint无query smoke一次通过：`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`、`checkpoint_load_strict=true`、`backward_count=3`、`trainable_fraction=0.005172846819097263`、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`、`performance_result=null`。
- smoke完成时最高状态为`LANDED`；随后只运行了一次正式15-row truth-blind prediction矩阵。

## Target5最终结果

- 正式prediction矩阵约9秒自然完成，未使用后台常驻进程、未重启。矩阵receipt为`PREDICTIONS_COMPLETE`；15／15行均有非空`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`和`receipt.json`，矩阵级`truth_opened=false`、`source_opened=false`。
- 15／15行均严格加载正式bundle，先完成3次support反向传播后才打开query；`query_opened_before_adaptation=false`、`query_role_opened=false`、`query_truth_opened=false`、`query_state_update_count=0`、`source_opened=false`，可训练比例均为0.517285%。
- prediction全部闭合后才连接三份既有同row truth sidecar；15个`score.json`和`target5_summary.json`均非空，无scorer错误。

|场景|DA0_REG0旧类均值|DA1_REG0旧类均值|DA0_REG0 floor|DA1_REG0 floor|均值变化|floor变化|决策变化数|
|---|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|67.50%|67.50%|35.00%|35.00%|0.00pp|0.00pp|1|
|`leo_low_elev_weak`|61.67%|60.83%|35.00%|35.00%|-0.83pp|0.00pp|5|
|`leo_rain_weak`|64.17%|64.17%|45.00%|45.00%|0.00pp|0.00pp|0|

- 15行中6行各有1个query类别决策变化；适配前后最大绝对余弦分数变化范围为0.000752628～0.054594249。rank-8已经能够越过部分判决边界，但5个low-elev变化产生净负收益，clear的1个变化净效应为0。
- 聚合结果为`mean_delta_pp=-0.2777777778`、`floor_delta_pp=0.0`，未达到+1.0pp／+0.5pp门槛，最终结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`；严格不进入Target25。
- 科学解释：从rank-4到rank-8，容量增加使分数扰动和决策变化显著增大，但更新方向没有对齐目标域旧类判决边界，尤其在low-elev场景出现一致负迁移。因此“fusion容量不足”不能单独解释旧候选的零收益；下一候选不应继续无约束放大同一方向，而应在仍低于1%的少层预算内改变可适配表征位置或约束更新方向。
