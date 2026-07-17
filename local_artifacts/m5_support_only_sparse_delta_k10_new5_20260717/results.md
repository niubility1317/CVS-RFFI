# M5 support-only sparse key-layer delta

## 结论

三场景support-only LOO统一锁定`B_input_proj`。该路线满足项目.md第281–288行的Stage2-C窄例外和全部资源上限，但formal query性能没有超过完全冻结的identity-only top1，因此只能保留为负向开发证据，不进入125确认矩阵。

## Support-only白名单选择

所有候选使用完全相同的超参数：5epoch、5个full-support optimizer step、SGD、lr=0.001、momentum=0；损失为leave-one-out prototype CE+旧类pairwise retention+新类separation。选择键为三场景support LOO的`min(old floor,new floor)`、H、遗忘、状态大小，query没有参与选择。

|候选|精确白名单概述|更新原参数|support old|support new|support H|old floor|new floor|support遗忘|FP16 patch|patch+head|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A_tail_idproj|`model.id_backbone.cls_head.id_proj.0.{weight,bias}`|25,760|77.78%|78.00%|77.89%|36.67%|46.67%|9.44pp|51,520B|80,120B|
|B_input_proj|`t_proj+f_proj+freq_stats_proj.0+pa_stats_proj.0`精确weight/bias|22,080|77.78%|77.33%|77.55%|40.00%|46.67%|9.44pp|44,160B|72,760B|
|C_tail_gate|`model.id_backbone.cls_head.id_gate.0.{weight,bias}`|25,760|77.22%|78.00%|77.61%|36.67%|46.67%|10.00pp|51,520B|80,120B|

## 预锁定B的formal query结果

|场景|注册前old|注册后old|seen-new|H|old floor|new floor|遗忘|
|---|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|91.67%|84.17%|87.00%|85.56%|70.00%|70.00%|7.50pp|
|leo_low_elev_weak|75.00%|65.83%|74.00%|69.68%|40.00%|60.00%|9.17pp|
|leo_rain_weak|81.67%|68.33%|68.00%|68.17%|40.00%|55.00%|13.33pp|
|三场景pooled|82.78%|72.78%|76.33%|74.51%|51.67%|61.67%|10.00pp|

逐类结果保存在`results.json`的`query_score.scenes[].after.per_class_index`和`query_score.pooled_after.per_class_index`。

## 资源审计

|项目|B_input_proj|
|---|---:|
|更新的原checkpoint参数|22,080|
|epoch/optimizer step|5/5（每次适配）|
|optimizer|SGD，momentum=0|
|optimizer状态持久化|否|
|FP16 delta patch payload|44,160B|
|注册后int8 support KNN head|28,600B|
|patch+head|72,760B（低于262,144B）|
|注册后dot MAC/query|28,160|
|patch合并后部署新增MAC|0|
|训练updated-layer MAC估算|注册前19,872,000；注册后36,432,000（每场景）|
|B三场景注册前后训练总时长|14.93s|
|单次训练最长时长|2.68s|
|峰值额外GPU分配显存|69,401,600B|

## 对identity-only top1的Pareto变化

|方法|注册前old|注册后old|new|H|old/new floor|遗忘|状态|dot MAC/query|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|identity-only top1|83.06%|72.78%|76.67%|74.67%|51.67%/61.67%|10.28pp|28,600B|28,160|
|M5 B_input_proj|82.78%|72.78%|76.33%|74.51%|51.67%/61.67%|10.00pp|72,760B|28,160|

M5仅将遗忘改善0.28pp，却使状态增加44,160B，同时new和H分别下降0.33pp/0.16pp；不是Pareto改进。low-elevation与rain旧类floor仍只有40%，远低于项目目标。

## 协议时序

脚本先完成三候选×三场景×注册前后support训练与LOO，写入`selector_lock.json`并预锁定B后才首次打开query。A/C从未运行query；query只对预锁定B执行一次隔离测试，不用于训练、适配、校准、选择、早停、回滚、候选排名或后续调参。测试时先在完全不读取truth sidecar的predictor阶段封存全部3个场景prediction artifact；只有三文件全部关闭后，独立scorer阶段才读取truth sidecar并重载prediction artifact一次性评分。每条query均在所有已注册类间独立决策，无角色Oracle、类别数量、quota或全批分配。
