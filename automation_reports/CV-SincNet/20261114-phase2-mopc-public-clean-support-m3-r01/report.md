# MoPC 仓库版 clean support 数值诊断

按用户要求排查 MoPC 并使用空闲 GPU；本 run 为源权重与当前 support 诊断，不读取 query 或 truth、不生成预测、不评分、不修改科学配方。使用四个已冻结的作者 attention ResNet 源模型中与指定 seed 对应的模型，来源 `20261101-phase2-mopc-author-group1-m4-r04/source`。A 训练配置为固定 200 epoch；旧 source recipe 记录的 paper 只描述当时 C 的选择，A 未执行 C，加载时仅以这一字段允许当前 public_code，并核验其他 recipe 与完整 source contract。

三条独立诊断：P01/MIX01/1-1、seed 2026092702、K10/N20，GPU5；P02/MIX05/20-1、seed 2026092702、K10/N20，GPU6；P03/MIX01/7-7、seed 2026092703、K1/N20，GPU7。clean 数据取对应三池同物理记录的配对地面 IQ，只索引各 split 当前新类 support。沿用仓库版 public_code 目标及原 20 epoch/阶段、5 类/阶段、batch 16、SGD 0.01/momentum 0.9/weight decay 0.0002，不启用梯度裁剪。源数据先做真实 checkpoint no-query smoke；loss/梯度非有限即原 guard 失败，保留日志。物理卡实际空闲和 6000 MiB 启动余量在启动前核实，其他 owner 不干扰。

输入为 `preparations/20261114-phase2-clean-non-issl-full-m4-r02/runtime/native.json`、clean bank 和上述 source；命令为 `python -m experiments.pamf_phase2_practical.clean_mopc_diagnostic --runtime <runtime> --source-root <source_run> --capsule <capsule> --seed <seed> --k <K> --n <N> --device cuda:0`；各卡以 `CUDA_VISIBLE_DEVICES` 绑定，每条独立日志和进程。运行代码以本报告所属提交为准。旧 r15 的真实非有限发生于论文语义和另一输入，不能凭本诊断完成宣称旧故障修复；本诊断也不是正式对比预测。
