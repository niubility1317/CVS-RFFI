# Time-only Rank-8 Prototype-Aligned Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`b9e289d17490bc6b5892334ddcfc3ceee104f89a`；首次push后独立回读远端分支OID一致。
- Phase1：`phase1_adv3b02_meta_adapter_time_r8_proto16_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 候选保持time-only rank8、P4 FOMAML+Meta-SGD和正式3步support更新；Phase1及Phase2均在冻结原型余弦CE空间使用`support_logit_scale=16.0`，最终判决仍为同一冻结原型余弦argmax。
- 正式bundle严格回读预算5458／1055125=0.517285%；10个可训练张量只含原编码器`id/dom_backbone.meta_adapter_time`，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`；receiver=`20-1`；operating point为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；每点三类LEO weak，共15个row。
- 配置：`configs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.json`。相对已闭合time-only rank8 Target5计划，15个row完全不变，只替换`bundle_id`、`checkpoint_path`和`prototype_path`。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`固定received IQ及原capsule／split；候选变化不触发数据重验。
- Phase2仅读取合法target support IQ和support标签、正式bundle、冻结原型；不读取source／clean样本、source cache、query真值或query角色。
- query只在3步support反向传播结束并冻结模型后逐样本推理，不更新模型、原型、归一化统计或其他状态。
- 同一冻结判决规则比较`DA0_REG0`与`DA1_REG0`；REG0新类指标为N/A，不引入D92式分类头。

## N607预登记

- 账户：普通`N607`用户`szu2070436088`；环境：现有`CVS-RFFI`；GPU：0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/checkout`
- Target工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.out`
- 工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1.json --output-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1`。
- smoke命令：`python code/scripts/smoke_stage2_meta_adapter_no_query.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/smoke_config_no_query.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_smoke --device cuda`。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1/matrix_config.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1 --device cuda`。
- expected artifacts：`factory_receipt.json`、`smoke_receipt.json`、15个row各自的`predictions_DA0_REG0.npz`、`predictions_DA1_REG0.npz`、`receipt.json`和truth-last `score.json`，矩阵级`matrix_receipt.json`和`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误、launcher-wide故障，或至少两个row出现同一确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与科学门槛

- prototype-aligned候选唯一一次独立P0/P1审查结论为P0无、P1无；Stage2仅复用已验证factory／runner／scorer，不进行重复审查。
- 15个truth-free prediction row全部闭合后，才由独立scorer连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一少层候选。

## N607 release、工厂与真实checkpoint smoke

- 最终release提交：`dd743fe532f001582ae37bb38dd3df2a5a637813`；远端分支OID独立回读一致。release归档本地／远端唯一一次SHA256均为`feedb93c0ff13a78436814b004e68c9b0267541b75d43ab3235591bce30a16fc`，run专属checkout内16个相关入口远端编译通过。
- 发布前独立确认release、工厂、smoke、prediction和stdout目标均不存在，同名Python进程不存在；正式Phase1 bundle和冻结原型非空，GPU0空闲。
- Target工厂一次完成15／15个truth-free row，状态`TARGET_INPUTS_COMPLETE`；`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。
- 首次smoke在读取配置前因工厂未自动生成`smoke_config_no_query.json`而`FileNotFoundError`，未加载模型、未访问query、未产生性能结果。随后仅从首row配置精确删除`query_path`，本地断言其余字段完全不变且无query／truth／role／source／clean键后同步。
- 修复输入后真实smoke通过：`REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS`，严格checkpoint回读，3次真实反向传播，`adaptation_objective=frozen_prototype_cosine_ce_v1`、`support_logit_scale=16.0`、`trainable_fraction=0.005172846819097263`、`query_opened=false`、`source_opened=false`、`query_state_update_count=0`、`performance_result=null`。

## Target5最终结果

- 唯一正式prediction矩阵快速自然完成15／15行；矩阵级状态为`PREDICTIONS_COMPLETE`，`truth_opened=false`、`source_opened=false`。每行均严格加载正式bundle，执行3次support反向传播后才打开query；`query_opened_before_adaptation=false`、`query_role_opened=false`、`query_truth_opened=false`、`query_state_update_count=0`，objective／scale均为`frozen_prototype_cosine_ce_v1`／16.0。
- prediction闭合后才连接三个既有同row truth sidecar。15个`score.json`和`target5_summary.json`均非空，完整证据已下载到`E:\type10-7\local_artifacts\meta_adapter_recovery\target5_time_r8_proto16_p4_r1_complete_20260825`独立复算。

| 场景 | DA0_REG0旧类均值 | DA1_REG0旧类均值 | DA0_REG0 floor | DA1_REG0 floor | 均值变化 | floor变化 |
|---|---:|---:|---:|---:|---:|---:|
| `leo_clear_weak` | 63.3333% | 63.6667% | 35.0000% | 30.0000% | +0.3333pp | -5.0000pp |
| `leo_low_elev_weak` | 65.0000% | 65.0000% | 45.0000% | 45.0000% | 0.0000pp | 0.0000pp |
| `leo_rain_weak` | 65.0000% | 65.0000% | 45.0000% | 45.0000% | 0.0000pp | 0.0000pp |

- 15行合计52个类别决策变化；每行最大绝对余弦分数变化范围为0.0161863～0.6821303。聚合`mean_delta_pp=+0.1111111`、`floor_delta_pp=-5.0`，未达到+1.0pp／+0.5pp双门槛，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`，严格不进入Target25。
- 科学解释：相较旧scale1候选的0个决策变化，prototype对齐与scale16确实把support梯度放大到能够改变判决，但clear weak尾类出现过冲。下一候选保持同一time-only rank8和3步预算，只把训练／部署一致的scale降至8，检验能否保留有益决策变化同时消除floor崩塌。
