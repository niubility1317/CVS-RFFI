# Phase3 固定 PCD 的无标签簇关联与匿名注册（2026-10-31）

Run `20261031-phase3-pcd-cluster-registry-s2026092701-r01`。使用已选 PCD 表征推进真实未来匿名注册，不继续只改身份头。原 signal/HIER 注册已显示 append-only 的固定旧 margin 不能提升同一后来群体的 R_U 或纠正旧类身份。本行改为全 U 原型判决及源支持回退，算法和参数在本次查询前固定，旧结果均保留。

## 冻结表示与新状态

严格复用 fresh f4b5/r01 的 TARGET_PCD444、原 F0/default6、DURA v1.0 R69、PCD known90/OE32/whitener，以及原六行阈值矩阵中的 PCD 两行（每行六阈值）。网络不更新、原 GMM/阈值不重拟合；PCD 有 U576 神经祖先，不是 source-only。先 actual F0/r03/G/PCD synthetic no-query smoke，再使用原 L6300 的最终 PCD z160 计算源统计；不能使用训练缓存中的 parent_z 冒充最终 PCD。

在原固定白化坐标下，用合法 L 标签建立六个均值，源支持半径 r_s 为 L 到自身类中心欧氏距离的总体95%分位，正数floor1e−6。该统计只用源 L，不读 OE/V/NV/U 真值，原源类中心不是另训神经头。U576 使用既有 views[0] P1 和当前最终 PCD 再前向一次，F0 同 P1 给出原生 closed；不重抽 received 或重做实际 DURA。

全 U 按 opaque ID 排序执行与旧 HIER 相同固定 [HDBSCAN 参数](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.HDBSCAN.html)：min_cluster_size16/min_samples8、euclidean/brute、eom、n_jobs1/alpha1/epsilon0/allow_single_clusterFalse/copyTrue/max_cluster_sizeNone。簇数不预设，保留所有非噪声簇。若簇内原六类 closed 的最高票占比≥0.8，且原 predicted_ratio margin 的中位数≥0，则关联该预测旧类；否则获得独立匿名槽。0.8 是本行事前固定一致性门槛，不来自目标标签或真实 K。

每簇均值和半径来自全部成员；r_j=min(r_s,该簇成员到均值距离95%分位)，同floor。原型包含旧类关联和匿名两类，匿名名按最小opaque ID规范排序，不携带真实TX。预测先选最小距离/r_j 的受支持原型（比值≤1）；命中则输出其旧类或匿名标签。无原型支持时，只有旧margin≥0且到其F0预测源类中心距离≤r_s才接受旧类，否则拒识。该规则能改判旧类，也能拒绝原来接受的无支持样本，不使用旧max(old,new)接受下界。所有计算逐query独立。

## 数据、矩阵与评分

仍用同一 RX3-2/index34、03-01 U576；未来03-15/23 Q10238，旧4508/可发现新4320/仅未来未知7-11为1410。仅未来未知从U整类排除。源L/OE/V/NV和Phase1角色不改，source seenRX、crossRX clean、crossRX star全经原DURA。source V27000/NV3150与两Q复用f4b5完整冻结logits/z/原PCD margin，不重读IQ或用它们拟合注册状态，clean不学习。该Q已曝光，非新的独立盲测。

四读出pooled_REG0/predicted_REG0/pooled_REG1/predicted_REG1，唯一主predicted_REG1；REG0精确复制PCD两列。REG1共用同一U簇状态，分别使用旧pooled/predicted margin回退。注册状态在任何V/NV/Q数值读取前冻结，全部4×3预测完后CPU scorer读U及sourceNV/clean/star四truth各一次；U真值不进入任何模型或注册状态。

新对齐采用U-only严格多数新类单元的一对一Hungarian，已知簇不匹配、旧类不重命名。未匹配匿名接受是错误身份，不算拒识；未来三可发现设备全部进入A_N，包括未发现者0。源NV19-15/19-16/7-1与目标5-5/10-7/2-19不同，source只报旧类与其全部未注册NV拒识，A_N/H_KN/H_OW=N/A。未来REG0对全四非旧类报R_U，REG1主R_U对固定future-only类9；另报相同later群体REG0/1配对及未成功注册真实类集合，不能把分母变化写成提升。

主报宏A_K/R_U/H、C_K、A_N和H_KN/H_OW；旧类正确接收相对同DURA原F0下降≤2pp、注册前后下降≤2pp，旧及预期注册集合C_K≥90%、总体A_N≥80%、注册后later R_U≥95%，与sourceknown12/必需景完整覆盖同时验收。A_N 的 80% 门槛沿用总体口径，各景完整报告；旧类与拒识保护同时检查总体和必需主景。原REG0仍按cleanRU95/starRU90。03-23缺两旧类的完整宏AK/CK/H保持N/A，later RU独立可报。零簇/漏发现不能绕过完整A_N分母。未全过不晋级/扩seed，完整目标ACTIVE。

## 执行与预算

唯一owner root，新的不可覆盖root/代码commit、聚焦负测及一次直接P0P1后登记；N607一次preflight/archiveSHA/compile，依次producer→独立CPUscore。source L新PCD前向6300、U新PCD576/F0576，真实signal6876/F0576/G0，原Q及源V/NV前向0；合成检查独列。新统计是源支持均值/半径、U HDBSCAN和K个全簇支持原型，无NN/新GMM/τ优化。部署继承834665神经参数，另六源中心和K簇中心/半径/标签，source/U逐行bank不部署。

本行只将U真值单独传入新release的score_truth路径，供预测全冻后的独立scorer；未来truth沿用既有路径，不传给producer。每GPU新增≤2、本行顺序worker且实际free≥4096MiB，不停止他人。非法来源/角色/状态改变/nonfinite/输出碰撞或不能闭合法预测即保留停止，低科学分数仍完成全部矩阵。不依据Q重选半径、簇阈值或读出；正式RX24-18不访问，结果不替代旧RX24-1困难集合。
