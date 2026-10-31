# PAMF 目标原型开放投影适配

VERIFIED / ANALYZED，两臂888更新、六列×三DURA预测与独立三truth评分全部完成。科学结论 NO_PROMOTION，完整目标 ACTIVE。

Run `20261031-phase3-target-prototype-s2026092701-r01`，actualcode `35d41680b2039452562baa76d407e7d0ef0dacdb`。两独立训练臂SOURCE/TARGET_PCD各444；六列×三DURA唯一主PCD_predicted_ratio，固定公式、数据、预算、科学门槛和停止规则见 `docs/PAMF_PHASE3_TARGET_PROTOTYPE_PLAN_20261031.md`，逐row/六类seed在experiment.json唯一维护。

N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-target-prototype-s2026092701-r01-release`，owner `codex-phase3-target-prototype-20261031`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_target_prototype_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-target-prototype-s2026092701-r01-release/runtime.json`。不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-target-prototype-s2026092701-r01`。

Source-only r03初始化，冻结232033编码器，仅训练102688投影；TARGET另2560aux原型（K16事前容量非trueK）。source原crossset/OE+KD10，两armsource索引匹配；TARGET+.1(global+local)平衡原型强弱view，source控制在U打开前完成。原F0/defaulthead/G/r03不改，SOURCE无U祖先、TARGET及源聚合有U祖先；无V/NV/Q神经训练。两arm全冻后各sourceGMM和V四new6tau，old4af两old6tau exact，全部τ冻后NV和两Q全6x3预测，再独立CPU3truth/Utruth0，source科学失败仍全部Q。原基线和native-energy延后状态保持，无发现/匿名REG，AN/HOWNA。

## 已完成结果

唯一主PCD_predicted_ratio宏AK/CK/RU/H：source_DURA: 89.740741/96.814815/64.571429/75.103446%；clean_DURA: 70.312500/86.805556/100.000000/82.568807%；star_DURA: 64.930556/86.631944/76.041667/70.048235%。同路径原F0闭集损1.240741/9.722222/7.986111个百分点。

对原E0pred星地正确接收8增3失/net+5，未知拒对2增5失/net−3，接收net+6；clean正确net−3/unknown0，source正确−1/unknown+83。对同预算SOURCEpred，星地正确6增10失/net−4、未知13增1失/net+12；SOURCE自身starAK65.625/RU71.875，不能以其较低拒识控制或主H微增替代旧基线/联合门槛。sourceknown12通过，source42/目标C90R90/2pp/coverage未满足，不换主、不依Q调权重/容量/预算，保留原predicted_ratio与pooled对照。

SOURCE/TARGET各完整444步与源索引匹配，SOURCE无U且在U打开前完成；编码器/原F0/defaulthead/G/r03/buffers均不变，只有102688投影各更新444、TARGET另2560aux原型。SOURCE保持source-only，TARGET及其新sourceGMM有U祖先；V仅校准4组六τ、旧两组精确复用，全部36τ冻结后才NV/Q，Utruth0。source parentcache12434、U三view1152/G384、学生projection454656行，eval及祖先成本另列；aux/逐行训练bank不部署，不能称匿名注册或严格未见全部未知。

708files/data0单archiveSHA/compile/runtime VERIFIED。driver1375698/train1375719、predict1377133独立live/CWD/argv/CUDA1，GPU训练初段20/406MiB、预测446MiB为离散样本非peak；CPU1378094仅pipeline/gpuNone/代码CUDAempty/产物，未独立live。终态owned空foreign0。23测试例+25子项/workerAST8/2CLI+launcherimport/一次P0P1通过；完整888步曲线、18点和分组见`docs/PAMF_PHASE3_TARGET_PROTOTYPE_RESULTS_20261031.md`。Git登记前共享HEAD并发冲突已在发布前无破坏恢复，实际三阶段均returncode0，不列为模型技术故障。
