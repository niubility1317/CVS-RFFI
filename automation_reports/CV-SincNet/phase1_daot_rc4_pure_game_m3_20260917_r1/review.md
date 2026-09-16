# 启动前定点验收

- 独立审查者native_parity_audit：PASS，新增21行未发现P0/P1。
- 6条pure_game真实CUDA步骤及checkpoint重建PASS；将RC4校准、路由、RC4 loss与DAOT loss函数替换为异常桩，全部未被调用。
- 原生F0对应AMP/FP32教师、RC4路由、BN与RNG检查PASS；E1/E21/E61/E161步骤有限；外部checkpoint读取0、target输入0。
- 响应求解器、复审回归及原生四格相关pytest全部通过。
- 21行实际训练parser通过，source/target划分与scratch参数检查PASS。
- 发布前核对旧dispatcher无待派发行，维持本轮唯一新派发owner；每GPU最多2个训练实验。输出根目录已存在即拒绝，禁止复用或覆盖。

完整证据：本目录config_acceptance.json、local_acceptance.json、native_acceptance.json；实际远端发布与启动证据在landing与launch_readback.json中。
