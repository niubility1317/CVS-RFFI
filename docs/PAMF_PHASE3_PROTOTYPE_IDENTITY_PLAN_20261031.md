# Phase3 源原型身份与 PCD 条件拒识（2026-10-31）

Run `20261031-phase3-prototype-identity-s2026092701-r01`。源端簇几何排障显示 PCD 单位坐标的跨接收机最近源类中心闭集诊断较强，而无标签簇与跨日支持尚未通过筛选。本行检验不依赖目标簇的源原型身份判决，不直接晋级失败的聚类方案。原 PCD 研究基线及所有历史 NO_PROMOTION 保持。

## 固定方法和顺序

严格复用 fresh f4b5 的 TARGET_PCD444、原 F0/default6、DURA v1.0 R69、PCD known90/OE32/origin/whitener 和旧两组六阈值。新神经、GMM、DURA、目标适应更新均0。先实际 checkpoint 合成烟测，通过即继续；本行真实 IQ 前向0，source V/NV/Q 复用已冻结的相同 DURA 特征与原 F0 logits。

从 source-cluster-geometry/074a84c 的 source_features.npz 精确复用最终 PCD 单位坐标 L6300，以同 IDs 对齐原 L 的合法 y/rx/day 元数据。六类中心是各类单位向量的算术均值，不再单位化中心，与源端已核实的最近中心规则一致。其身份 c=argmin_j ||unit(z)−mean_j||；只使用六个已注册源类，不读取目标标签、真实未知类数或类别配额。

六中心先于 V 数值冻结。已知 V27000 用该原型 c 路由 PCD 原两种密度分数：pooled_ratio 与 predicted_ratio；后者取 c 的15个源RX/day模式减 OE32 背景。两种分数均由 c 索引各自六个新阈值。固定复用 star_macro12_risk.calibrate_known，以新 closed、原 F0 baseline_closed 和已知 V 标签校准；硬保护仍是总体＋五源RX＋六场景共known12，30交叉组另报。新原型闭集较强也不能替换原 F0 的2个百分点保护基准。

两个新六τ完全冻结后才读取源 NV 与两个未来 Q。所有4读出×3条件预测完成后，独立 CPU scorer 才连接 source NV、clean Q、star Q 三份真值各一次，U真值0。原 F0 logits 与原型距离证据分开保存；新 closed 不来自原 F0 argmax，不能称原身份判决不变或把距离冒充神经 logits。

## 数据、矩阵与指标

source L6300/OE6134/V27000/NV3150、五源 RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23和六旧TX均保持。新读 L 特征/合法标签6300，原 Us56700及OE逐行值0；V仍是额外有标签校准角色，不是独立已知类测试。

目标仍为开发 RX3-2/index34、03-15/23未来10238条，旧4508、未注册四类5730；前期 U576 仅为所用 PCD 的继承适应来源，本行不新读U。六旧类为14-10/14-7/20-15/20-19/6-15/8-20，目标非旧为5-5/10-7/2-19/7-11，四类本行均保持未注册。三个测试条件为源已知RX星地、跨RX clean、跨RX星地，均已经过同一冻结DURA；clean不学习，未来同canonical清洁/星地记录不重复算独立样本。

四读出固定 OLD_PCD_pooled_ratio、OLD_PCD_predicted_ratio、PROTO_pooled_ratio、PROTO_predicted_ratio，唯一主为最后一项。旧两列复用 fresh PCD 原预测及阈值，不重拟合；两个新列只改变源原型身份、分数路由及已知V新τ。主报逐TX宏A_K/C_K/R_U/H、原F0同DURA闭集与新原型闭集、原F0闭集−A_K，完整TX/RX/day/六景和配对得失。

source known12保护、目标总体及必需主景的clean R_U≥95%/星地R_U≥90%、C_K≥90%、原F0−A_K≤2个百分点、必需类别完整覆盖仍需同时通过；主星地 A_K须高于 OLD_PCD_predicted_ratio，R_U不得下降。未全过NO_PROMOTION、不扩seed、不降低门槛；03-23缺两个旧TX时完整宏指标N/A。匿名注册未启用，A_N/H_OW=N/A。此为复合判决变化，不是新UDA训练；PCD有U576祖先，不称纯源模型或新增适应成功。

## 执行与成本

EXACT6公开配置为run_id/code_commit/reference_config/source_geometry_root/output_root/batch_size128，不含truth路径。唯一owner root，不可覆盖新root，N607单predict GPU worker再独立CPU scorer；每GPU新增≤2且实际free≥4096MiB，不改变其他用户任务。

真实F0/signal/G新前向均0；checkpoint合成烟测单列。新统计为六中心一组（6×160 FP64=7680 B）与2×6新τ，原GMM/旧τ保持。运行比较可加载F0/r03/PCD/G共1169386参数；主算法最小神经路径为PCD334721＋G266951=601672，F0仍用于本次原始基准/控制，不能把缓存预测时间当新部署B1时延。

角色/祖先不符、冻结状态改变、nonfinite、输出碰撞或无法产生合法预测即保留停止；低科学分数仍完成三条件全部预测和评分。代码commit、聚焦负测、一次直接P0P1、最小登记、单N607preflight/archiveSHA/compile、PID/CWD/argv/GPU/log增长后按实际结果归档。Q已曝光、单seed/开发RX、global packet history UNKNOWN及正式RX24-18未访问的限制保持，完整目标ACTIVE。
