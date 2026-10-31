# MILKR 原始细节保留编码实验

R28 固定 Tri-SIG-TIME SSL、DURA v1.0、原神经分类头、C4 提取、R27 当前 support 置信度先验及 R26 query 融合规则。新假设是共同空间可能压掉 raw 中有用的身份细节，因此每个 block 保留 64 维 CCA，同时加入 source PCA 的前 d−64 维；两分量各自单位化、各占一半能量，拼接后仍是原 d 维。原生分量使用 raw/G 源样本联合学习的同一个均值和 PCA 基，跨路径不另选坐标系。

备选为继续调先验或只改变 CCA 相关性权重；本轮选择直接保留未被共同空间表示的信息，避免继续只改变旧/新分数平衡。权重 .5、CCA64、PCA d−64 均在运行前固定，不用目标结果筛选这些值。源编码器参数增加，运行成本实际报告，不假定加速。

source L6300、既有 ep91/92，共12600 raw/G 配对；五折每折同排除 held RX 和两个 TX，CCA/PCA 均只用其余 source 行。候选与 CCA-only 对照各用同一120核/正则规则，K1选 Q0/QF 四路径宏H，K20保持当前 support 真实物理三折 OOF。源主路径宏H平均增益须≥0.5个百分点，且每个源 K1/20、S0/S1 的 raw 宏H差不得低于−0.25个百分点；失败就 source-only 停止，target0，不训练最终编码器。通过后才拟合全source参数并进入目标。原F已见源六TX、源三景/目标六景和old4new2代理边界明确，不称新盲测。源held新类head列先剔除再softmax。

只把均值偏置/投影作为编码器权重存储，target无source样本、类条件统计或协方差库。support仍严格S0raw/S1G，query Q0raw/QG仅DURA/QF同记录融合。所有置信度、带宽和KRR参数仅当前SUP，OOF各折仅训练SUP。F/G/head冻结；query不参与拟合或批统计，G每原received执行一次后再C4。

源通过后首RX1-1/P01/MIX01/seed2026092701、K1/20×N2/20，三query B12+C24共36行全部预测固定，独立truth-last评分。每角点两S raw较匹配原raw≥5个百分点、QF>raw，且raw旧新均值guard≥原版−1个百分点才扩其余12KN；全部16格通过后补另六RX。最终七RX等权逐KN验收，原完整目标不缩小。QG匹配控制完整保留，QF与原QG为输入改变的差。

sourceGPU4/targetGPU0，启动前核实容量。代码commit、环境/CWD、命令、来源/data句柄、输出及技术停止规则见experiment.json；实际checkpoint无query smoke为launcher首步，PASS直接继续。独立P0/P1审查、单包传输/remotecompile、所属PID/CWD/argv/GPU/log增长和独立truth评分保持。

## 发布前核验

21项相关检查与4项非法协议输入检查通过，修改文件本地compile与diff检查通过。唯一独立P0/P1审查PASS：CCA/native PCA各半能量、维度/保存/加载一致；source候选及对照独立校准；源门槛失败不生成最终模型或启动目标；prior/gate/fusion保持原路径。实际F/G无query smoke将在launcher首步执行，PASS直接继续。
