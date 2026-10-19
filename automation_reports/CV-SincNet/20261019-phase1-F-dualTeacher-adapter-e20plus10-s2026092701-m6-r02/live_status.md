# 实际启动后态

运行 r02：RUNNING，code/config commit `9a376ab313b3ba42459df32694c1502675e02a59`，归档 transfer 与 remote compile 均 VERIFIED，远端 78 个聚焦负例 PASS。首批 3 行真实父权重 no-query smoke PASS，随后实际完成第 1 个追加 epoch（累计 E21）。

|行|GPU|PID|每轮 L 学生包|每轮 U 学生包|每轮 own EMA 包|每位教师包|冻结检查|
|---|---:|---:|---:|---:|---:|---:|---|
|NOAS_FT-s2026092701|4|2249313|28416|56700|56700|28416|PASS|
|NOAS_ADAPTER-s2026092701|6|2250029|28416|56700|56700|28416|PASS|
|S0-s2026092701|7|2250112|56832|56700|56700|28416|PASS|

独立核对 PID/CWD/cmdline、GPU 实际进程、CUDA4/6/7、CPU2、FP32/TF32off 与增长的 step 日志。NOAS_ADAPTER 实际 trainable4768/total905490；FT 两行 trainable231233/total900722，分类头均排除。适配器行原 F 的190个参数冻结核对 PASS。

S1/S2/S3 仍按原容量规则排队。此处是训练后态，不是最终性能或完整矩阵。自动流程将执行 fixed FINAL30、固定源资格、七视图预测及独立 truth-last 评分。源或正式结果未达到目标不触发技术重试。

证据：local_artifacts/F_dualTeacher_adapter_E30_20261019/r02_progress003.json；r01 失败另见其 final_analysis.md，全部旧产物保留。
