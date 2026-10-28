# PAMF Phase2 Full eq 数据分配 r08

2026-10-29。用户已授权完整数据上传与实验数据制作；本轮不启动身份训练、身份预测或评分。主线保留最初提出的八套物理独立综合数据，增加 query 总量，复用原 source 数据与 Phase1 权重。

## 1. 为什么调整 r07

实际 Full eq 已全量验证：20 个 ZIP、158 个 PKL、8,767,867 条 IQ。数据不均衡，不能将这个总量平均分给每个 TX/RX/date。

八套综合＋三个纯场景若每组均用24条 support 候选和500条 query，需要每 RX/TX 有5,764条；即使旧六类，最少四日也只有4,404条，且还必须保护历史 query 角色并排除已有 Phase2 观测。因此，这个规格不能直接执行。

r07 的单个综合池500条 query、8次 support 重复可以减少用量，但八次并不是八套独立测试数据。r08 优先保留八个物理独立综合组，每组每类每 RX 为150条 query，八组合计1,200条；三个纯场景继续各500条。比重复同一份500条 query 更能体现数据与信道采样波动，统计时仍须区分共同模型 seed 和独立物理组。

## 2. 原始记录、历史角色与选类

源侧保持六旧类、5个source RX、D1至D3、L_s6,300/U_s56,700/V27,000，U_s TX隐藏。已匹配的三视图PAMF、外部方法及冻结DURA权重身份不因Full扩容改变，也不追加Full源训练记录。

Full 的文件名提供 RX/day，`node_list` 提供 TX，`data[i]` 为float64的N×256×2。不能沿用compact默认RX/day/eq或把Full索引当作ManyTx索引。先在同TX/RX/date内逐点精确匹配原始float64记录，建立以下规则：

- 新support排除ManySig、ManyTx、ManyRx、SingleDay的全部记录并集，保守保护历史query。
- 新support和query共同排除整个ManyTx，覆盖已知历史Phase2 received的数据来源；原received、角色和结果保持。
- 其他旧compact记录可继续仅作query。此为明确命名的新Full基准，不能声称未接触过的独立盲测，也不声称已穷尽项目所有历史数据来源。
- 本轮11个物理组互斥，全部support/query角色互斥；一个物理记录只生成一次received。P0/P1、模型seed、K及support重复均不增加物理样本数。

源外20个新TX按排除后的最低四日可用量排序，继而按最低fresh support量、最低日期原始量及名字打破并列；要求7RX×4日非空，且满足本轮全部配额。最终20个覆盖名单及固定随机到达顺序由库存结果生成，K/N不按覆盖排序到达。只读计数和精确身份，不依据IQ质量、SNR、模型或准确率选类。原20新类名单与r07候选不静默改名。

日期分配先预留所有组的fresh support，再从剩余合法记录分配query；以最大最小均衡和整数配额尽量平衡四日。实际余量不能支持每组每类完全相同的日期比例，最终报告逐RX/TX/date配额并保留分日期评价。域内信道概率不依赖TX、旧新类或support/query。

## 3. 物理分配总表

|用途|独立物理组|每类／RX／组 support 候选|每类／RX／组 query|实际物理support|实际物理query|
|---|---:|---:|---:|---:|---:|
|综合主基准|8|24|150|34,944|218,400|
|low suburban|1|24|500|4,368|91,000|
|mid urban|1|24|500|4,368|91,000|
|low urban|1|24|500|4,368|91,000|
|总计|11|||48,048|491,400|

共539,448条物理记录。每个RX/TX总需求2,964条，其中264条fresh support、2,700条query。每组24条是候选池，某行方法只能接触实际K条，不能读取余下候选。

K为1/5/10/20，嵌套前缀；N_new为0/2/5/10/20。每个综合组用一个事前固定support seed，八组共八个独立数据＋support任务，不扩成8×8。每个纯场景做5次support重复，query固定，重复不计作新测试集。实际组合为7RX×(8＋3×5)×4K×5N＝3,220个split；7RX×11组×5N＝385个注册阶段capsule。

综合单组旧类query900条、N20时全部类query3,900条；同RX八组分别合计7,200与31,200条。纯场景对应3,000与13,000条。不同K用同一query，方法、model seed、P0/P1严格配对。

## 4. 信道与阶段读取

综合使用已固定六分量G10几何参考概率，环境先验各0.5；low总76.6257%、mid14.6359%、high8.7383%，不将其写成实测任务频率。单星被动监听设定不变。调用既有practical residual、post_sync、no additional EQ、25MHz；RX/day绑定receiver seed2027。纯场景分别保持原生low suburban、mid urban、low urban参数，不混为一个三分量场景。

P0保存固定received。P1应由同一received经冻结DURA得到，不重采样信道、不扩大K；本轮数据构建器不加载F/G，先制作P0与阶段读取接口，DURA计算缓存是后续明确的输入预处理任务，不能把它记为身份实验已经完成。

每个RX/物理组/N有独立query文件，query混合打乱且仅含当前已注册类。support文件按split存实际K条及可见标签。A使用query-only读取器，不读取target support；B/C显式加载当前行support和query。未注册类别、额外support候选、raw/clean/source路径、信道oracle和truth均不进入该行数据读取器。builder_only与score_only留在capsule外；未来集成身份方法时不得重新使用全量特征导出器提前读取所有未来类别。

## 5. 执行、复用与停止

数据run：`20261029-phase2-full-eq-preparation-s2026102901-r01`。配置为`configs/pamf_phase2_full_eq_data_r08_20261029.json`，登记及最小命令在同run的experiment.json/report.md。唯一controller按inventory→计数选类→allocation→received及capsule输出顺序执行，CPU单进程，不占GPU。

上传已逐包独立CRC校验，20个ZIP共34,378,471,511字节。原始ZIP只存一份，不按方法解压或复制。received只生成一次；query在同注册阶段的不同K、support重复和所有方法间复用。实际预算仍为23个数据/support单元，未增加源模型训练；外部方法原生训练预算、B/C机制和状态继承保持总实验表定义。

首次精确映射失败、fresh support不足、query不足、物理重叠、输出碰撞、非有限received或子进程错误即停止，保留partial，不覆盖或自动改参数重跑。builder不消费checkpoint，真实checkpoint无query smoke不适用于本数据阶段；后续身份运行仍须使用既有合法smoke及阶段权限。协议负测通过，独立P0/P1审查的发布成员类型问题已修复并定点确认。实际完成后读取manifest/catalog/计数和状态，不能仅凭提交PID宣布数据完成。
