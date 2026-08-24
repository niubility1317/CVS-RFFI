# FastTrust-QB3正式E200实施计划

1. 为有界域混淆、校准解耦、集合特征、最差receiver约束、分路预算、尾段退火、恢复诊断和五行矩阵添加聚焦行为测试，逐项观察预期RED。
2. 新增可复用的有界域混淆实现，并接入有标签与RC4无标签训练路径；保持旧`grl_ce`默认兼容。
3. 扩展RC4校准包和路由：P专用全局APS、集合安全特征、最差source receiver阈值、H/P独立class×receiver权重预算。
4. 将P-set和P-conditional拆分为独立系数及E181–E200尾段退火，增加稀疏梯度诊断、finite恢复checkpoint和首次异常包。
5. 增加C4冻结Core90无标签特征锚点，确保不读取U的TX truth；冻结C0–C4配置、worker参数和不可覆盖launcher。
6. 加速配置固定为一GPU一行、AMP/TF32、eval batch1024、source-heavy每10epoch且末20epoch逐epoch；验证不改变E200、U256和训练步数。
7. 运行聚焦GREEN、相邻回归、真实Core90无query smoke；完成一次独立P0/P1审查，若有直接问题只做一次定点修复和复审。
8. 创建最小预登记报告，精确stage本次文件，提交、自动push并核对远端OID。
9. 使用普通N607账号完成只读preflight；制作一次release归档，只比较一次本地/远端SHA，远端编译后启动矩阵。
10. 启动后只执行一次PID、CWD、cmdline、GPU映射和日志增长读回；状态报告为`RUNNING`，不提前声称性能完成。
