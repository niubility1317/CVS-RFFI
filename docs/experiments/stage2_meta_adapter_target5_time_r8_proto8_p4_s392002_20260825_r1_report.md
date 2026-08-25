# Time-only Rank-8 Prototype-Aligned Scale-8 Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`5043a79d871ed1d84a570c3f3d701bbb8ca3d5c8`；首次push后独立回读远端OID一致。
- Phase1：`phase1_adv3b02_meta_adapter_time_r8_proto8_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- 保持time-only rank8、P4 FOMAML+Meta-SGD、正式3步support更新及冻结原型余弦argmax；Phase1和Phase2共同使用`frozen_prototype_cosine_ce_v1`／scale8。
- 正式bundle预算5458／1055125=0.517285%；仅更新10个`id/dom_backbone.meta_adapter_time`张量，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`、receiver=`20-1`；`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`各三类LEO weak，共15行。
- 配置：`configs/stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1.json`。相对scale16 Target5计划只替换bundle ID、checkpoint和prototype，15个row完全不变。

## Phase2权限边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`、capsule和split，不重验数据。
- 仅读取合法target support IQ／标签、正式bundle和冻结原型；不读取source／clean样本、source cache、query真值或query角色。
- query在3步support反向传播结束并冻结状态后才逐样本推理，`query_state_update_count`必须为0。
- 同一冻结判决比较`DA0_REG0`与`DA1_REG0`；REG0新类指标为N/A。

## N607预登记

- 普通`N607`账户；现有`CVS-RFFI`环境；GPU0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1/checkout`
- 工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1`
- stdout：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1.out`
- expected artifacts：工厂／smoke receipt、15行DA0／DA1预测与receipt、truth-last `score.json`、矩阵receipt及`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误或重复确定性pre-prediction异常时停止；不得因低性能停止。

## 审查与门槛

- scale8候选唯一审查P0无、P1无；Stage2无代码变化，不重复审查。
- 15行prediction闭合后才连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25，否则科学失败且不晋级。

## Release、工厂与smoke

- 最终release提交：`a3ae8edc2e4cb8e57ca2014ce3377ba33ea2dc8a`；远端OID独立回读一致。归档本地／远端唯一一次SHA256均为`c143c8dda5fc955d67420b840c63ed1e3b54b377195a3843845bac5e4e51a664`，16个相关入口远端编译通过。
- 发布前确认6个专属目标和同名Python进程不存在，正式bundle／原型非空，GPU0空闲。
- 工厂一次完成15／15行，状态`TARGET_INPUTS_COMPLETE`，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。无query配置由首row精确删除`query_path`且本地断言其余字段不变。
- 真实smoke一次通过：3次反向传播，`frozen_prototype_cosine_ce_v1`／8.0，`trainable_fraction=0.005172846819097263`，`query_opened=false`、`source_opened=false`、`query_state_update_count=0`。

## 正式运行与truth-last评分

- 正式prediction自然完成15／15行，15份`DA0_REG0`、15份`DA1_REG0`、15份row receipt和矩阵receipt齐全；目标run无残留训练进程，日志未检出Traceback、RuntimeError、CUDA OOM或协议违规。
- prediction阶段`truth_opened=false`、`source_opened=false`；每行均为3次真实反向传播，query在适配前未打开、适配后状态更新计数为0。15行prediction完整后，独立scorer才连接truth；15份`score.json`均验证same-row ID一致，REG0新类指标保持N/A。
- 完整证据回读至`E:\type10-7\local_artifacts\meta_adapter_recovery\target5_time_r8_proto8_p4_r1_complete_20260825`，共65个文件；独立分析脚本逐行复算并通过全部协议、预算和artifact断言。

| 场景 | DA0_REG0旧类均值 | DA1_REG0旧类均值 | 均值变化 | DA0_REG0 floor | DA1_REG0 floor | floor变化 |
|---|---:|---:|---:|---:|---:|---:|
| `leo_clear_weak` | 63.3333% | 63.5000% | +0.1667pp | 35.0000% | 35.0000% | 0pp |
| `leo_low_elev_weak` | 64.1667% | 63.5000% | -0.6667pp | 40.0000% | 40.0000% | 0pp |
| `leo_rain_weak` | 64.1667% | 64.1667% | 0pp | 45.0000% | 45.0000% | 0pp |
| 三场景聚合 | — | — | **-0.1667pp** | — | — | **0pp** |

## 结论与下一步

- scale8在15行中造成16个query判决变化，各row最大分数位移范围为0.002172～0.198973，证明少层梯度适配真实作用于判决，并非零更新。
- 但聚合旧类均值变化`-0.1667pp`、floor变化`0pp`，未达到预注册的`+1.0pp/+0.5pp`晋级门槛。结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`，不启动Target25。
- 与scale1零判决变化、scale16有52次判决变化但clear floor下降5pp合看，单纯缩放冻结原型CE只能调节更新幅度，不能稳定控制跨场景方向。下一候选应增加只依赖support的有界更新机制，而不是继续扩大固定scale。
