# qKNN极轻型5/10/20新类目标模式优化报告

本文件镜像`E:\type10-7\automation_reports\CV-SincNet\qknn_extreme_light_5_10_20_20260714_215758\report.md`，用于Git版本承载。任务状态：`DESIGN_AND_PROTOCOL_PREREGISTRATION`。

## 成功门槛

|指标|门槛|
|---|---:|
|`old_acc`|>=88%|
|`min_old_class_acc`|>=85%|
|5类`seen_new_acc`|>=90%|
|10类`seen_new_acc`|>=88%|
|20类`seen_new_acc`|>=83%|

正式确认覆盖5个target receiver、至少5个独立seed、3个正式LEO场景；使用嵌套真实ManyTx TX集合和开发后锁定的统一K。默认冻结ADV3B02，1-view，adapter参数不超过50k，适配不超过20epoch，无query图，持久状态不超过128KB。禁止query真实角色、类别quota、query标签拟合和跨K/seed/新类规模拼接最佳结果。

## 当前执行边界

根目录`项目.md`已先加入第10.3.1节；根目录不是Git仓库，本仓库通过`docs/cvs_stage2c_extreme_light_goal_20260714.md`承载协议增量。当前尚未launch，后续需先完成ManyTx覆盖审计、本地实现与`ssr-gpu`验证，再执行N607新鲜preflight、同步、launch、完整日志分析与独立确认。

## 2026-07-14 22:17更新

ManyTx覆盖审计、嵌套5/10/20真实TX清单、开发/确认seed、K候选和资源上限已预注册。新增support-only对角度量余弦头：最大26类6,938参数、约27.1KB FP32状态、20epoch、1-view、无backbone梯度、无query图、无role/quota Oracle。新增20类exporter和resume-safe smoke/dev/confirm matrix runner。本地`py_compile`、6项pytest、exporter语法/dry-run、36-row matrix dry-run和端到端runner smoke均PASS。

N607 22:17新鲜inventory显示8张GPU各有1个约470MiB的RIEI训练进程。本任务按每GPU最多2个训练实验的许可，只计划在GPU0/1/2各增加1个20新类feature export，不干预现有任务。首次远端输出根为`runs/cvs_qknnv42_extreme_light_20new_features_20260714`，日志根为`logs/cvs_qknnv42_extreme_light_20new_features_20260714`。
