# ProtoNet→MoPC 仿射候选多 B 支持验证

对已完成的六个 clean ProtoNet B（原 full 队列任务 0000、0012、0024、0036、0048、0060），分别加载各次真实 B-K10 和 B-K20、对应 clean old/new 支持集、固定原 C seed，在 GPU 5、6、7 仅诊断公开仓库 MoPC 无裁剪 N20。每个候选先验证全部旧支持样本 argmax 完全一致、中心化 logits 和 softmax 容差；不通过即拒绝。每卡顺序执行四项，总 12 项，预期 K10 各 320、K20 各 560 次更新；仅观测，不读取 query/truth、不生成预测。

使用 `20261115-phase2-mopc-proto-affine-support-m3-r01-release` 的 `clean_mopc_savedb_diagnostic --variant affine_bridge --n 20`；每卡最多一个进程，启动前余量至少 6000 MiB、避让其他 owner。六个 B 来源和实验结果均原样保留，输出独占 `runs/20261115-phase2-mopc-proto-affine-validation-m6-r02`。仅能验证采样支持集上的行为；即使通过也不能将改变 C 特征参数化的版本和已有旧桥接预测视为同一实现。
