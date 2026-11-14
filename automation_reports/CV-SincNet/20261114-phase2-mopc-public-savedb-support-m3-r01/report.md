# MoPC 仓库语义的真实保存 B 支持集检查

在不读取 query/truth、不生成预测的条件下，检查当前 public_code MoPC 跨 B 模型和高 K/N 的真实数值行为，继续定位旧 r15 的不稳定。GPU5 使用 r15 ProtoNet-CDA `protonet_cda-s2026092702-c0195` 的实际 r11 B-K20、原 received K20/N20 support；GPU6 和 GPU7 使用 r11 DADDA `dadda_sda-s2026092703-c0332` 的实际 B-K10/B-K20 和原 received K10/N20、K20/N20 support。对应数据、B 状态、C seed 和原训练预算保持，源 checkpoint no-query smoke；不修改/重发任何旧输出。GPU 每卡一个进程、启动前空余至少 6000 MiB，避让其他 owner。

入口为已验证 `group2_savedb_diagnostic`，代码 commit `8c744f4535e436dc9e173456162840cc2a1f966f`，使用当前 public_code/FP64 梯度观测及逐层仪表；r15 原故障是 paper/FP32，此组只能判断所测当前路径，不能证明旧故障修复。每项在独立日志记录最早观察到的非有限，原 guard 生效。
