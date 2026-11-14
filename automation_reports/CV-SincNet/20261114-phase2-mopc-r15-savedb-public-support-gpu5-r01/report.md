# r15 保存 B 的 MoPC 仓库版 GPU 诊断

只读复用 r15 任务 `protonet_cda-s2026092702-c0195` 的 r11 同次保存 B-K10、原 P02/MIX05/8-8 K10/N20 received support 和原 C seed，在空闲 GPU5 执行当前 public_code MoPC 注册，观测前向、buffer、梯度及优化器更新。无 query、truth、预测、评分或旧输出修改。原故障实际使用 paper 语义及 FP32 梯度范数观测，本次为 public_code 及 FP64 范数观测，目标、原型修正和停止路径不同；本次结果只能验证所测仓库语义的数值行为，不能当作原故障重现或修复证据；逐层观察会改变执行时序。原隔离及 partial 不重发。

入口 `python -m experiments.pamf_phase2_practical.group2_savedb_diagnostic --runtime <r15 release runtime.json> --job <r15 run jobs/00002.json> --k 10 --n 20 --device cuda:0`，以 `CUDA_VISIBLE_DEVICES=5` 运行，发布代码 commit `8c744f4535e436dc9e173456162840cc2a1f966f`。启动前检查其他 owner 和显存至少 6000 MiB，遇非有限按原 guard 停止并保留日志。
