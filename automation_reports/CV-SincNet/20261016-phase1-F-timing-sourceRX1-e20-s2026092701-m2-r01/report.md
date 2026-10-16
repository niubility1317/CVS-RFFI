# F 基座轻度分数延时增强：E20 源 RX1 留出

状态：LOCAL_VERIFIED，尚无新模型识别结果。唯一候选是假设性的有界采样落点变化；不能因算子检查、源健康或梯度通过就称为泛化有效。精确两行矩阵、seed、命令、路径和 stop rule 由 [experiment.json](experiment.json) 维护。

## 与另一个对话的分工

2026-10-16 已读取“开发 CVS Phase1 星地增强训练”（thread 01a13a98-fbec-7ba3-86b9-4cd56b8992c0）。其当时正在将 raw/raw、ordinary/ordinary、raw/ordinary 的监督组织迁到 F。本对话独立研究 IQ 分数延时，不重复旧 ordinary 组织、源谱响应银行、相位跨 TX 方向拟合或特征对齐。未向另一对话发送消息、启动其任务或改动其工作树。

研究代码从已提交的 F 工作分支 HEAD `24225c7be52edf8e2fd76b8dc7a88985d5a30a91` 分出；独立分支 `codex/cvs-iq-timing-e20-20261016`。E:/type10-7 根目录 .git 缺 HEAD，真实 Git 承载面及活动 worktree 已只读核对；没有重建根仓库或清理其他修改。F 代码身份为 A–F 执行提交 `09554c3a06c976c53efc4dbc61f5c501a1e7b06e`，复用代码，训练不继承其 E50 权重。

## 已有证据及不再重复的路线

这些是历史同 row 结果，不是本轮 F/E20 匹配基线。源 5580/720 与完整 L6300 分别标明；不同预算和配方的增益不相加。

|完整报告|实际范围与结果|本轮判断|
|---|---|---|
|20261014 sigtime-urban-paired E30 m18 r02|18 行、3 seed、正式 clean+6practical。普通 B0−C0：low +15.335 pp、clean −1.653 pp。课程 C2−C1 low +.619；间隔 C3−C2 +.107；新增 SSL C4−C3 +.170|主要收益来自实际星地输入；未达 low60，后续小增量未抵消 clean 代价|
|20261015 urban-ssl E50 m4 r01|4 行单 seed 完整评分；B0/C3/CONF/WEAK low53.9696/54.9351/55.0988/55.1839%；CONF−C3 low+.1637、clean−.2857 pp|不继续增加资格条件，不能用单 seed 小差宣布有效|
|20261015 branch-align E50 m8 r01|内部 L_train5580/L_dev720；T/F/P/R/S/J/ALL 七约束均源条件 FAIL；正式 low 全<60，ALL−A0 low−.0232/clean−.6137 pp|不继续强迫所有分支相同|
|20261015 joint160-kl-dir E50 m2 r01|完整 L6300；KL/direction low54.7577/54.8024%，direction−C3 low−.1327 pp|方向更近没有成为身份识别改善|
|20261015 semantic E50 m5 r02|内部5580/720；单向/双向排序 low53.4768/53.3542%，双向−单向−.1226 pp；各自 clean 仍低于无星地|不复用双向排序为新主线|
|20261015 bridge-teaching E50 m2 r01|内部5580/720；BRIDGE_SIG−CE low+.3012、clean+.0244 pp；双方都含共同 joint 一致性，两行 own EMA 校准数值不同；没有真正无一致性且全部匹配的 HQ 对照|不能单独归因 HQ 或用源 HQ 反事实成绩预测最终收益|
|20261015 originalSIGTIME-rdrop E20/E50 m18 r01 + 技术恢复 m4|18 逻辑行完整；E20 RDrop−ORIGINAL clean+.827183 pp（2/3），RDrop−TWO_CE+1.456746；E50分别+1.418056/+2.353968（3/3）|RDrop 是随机子网络正则，只有 clean 证据，不当 IQ Ag 或星地收益|
|20261015 SIGTIME-RDrop A–F E50 m18 r02|18 行、3 seed完整；F clean84.270238%±.275383，低于 A84.340079%；架构强化未获总体证明|按用户指定固定 F，不按此表换回高分 A，也不拿 E50 权重当 E20 父模型|
|20261015 mixedIQ E20 m5 r01|单 seed五行完整；轻度响应 Ag 源资格先 FAIL。B11−B01 low+.0393、clean+.0131 pp；clean交互−.1958 pp|该响应不是已证明的有效 Ag；不重复它或把微小目标正差晋级|
|20261016 ordinary 五源 RX 确认|原 SIG_TIME单 seed五折，mean+2.0296pp，但 RX6/8−.9259/−2.5926pp，原规则 FAIL|不把旧普通增强当已验证的域泛化|
|20261016 源相位 / finiteIQ 响应|相位跨 TX/day 重复性 FAIL；有限实测谱方向必要安全 PASS，但两 session 对旋转控制符号相反，尚无新训练|不从统计关联或安全 PASS 直接宣称训练收益|

上述报告均从实验总索引定位后读取完整相关结果；早期 RUNNING 标题由后文终态/逐行正式表纠正。未重训、重测或重评分历史模型。训练曲线结论沿用已有全日志报告；本轮不另作抽样曲线分析。

## 普通 IQ 实际执行审计

读取 N1-RX1 的 actual_native_args、ordinary_base_config、module_files 和真实 DataAugmentation/build_aug_base_cfg 调用。运行导入来自既有 evidence_responsibility_train20_r02 的 experiments/adv3b02_xuc/code，不凭根 code/ 默认值判断。普通算子作用于 dataset/native RMS 后的 L IQ；RDrop 两次前向使用同一实际增强结果。正常入口顺序为 DC→端部 taper→循环时移→增益→全局相位→CFO→相位噪声→多径→AWGN→DAC（关闭）→PA（开启）。

scale 真实参数为 min .10、max .35、warmup3、ramp15、curve1.25；概率乘 scale，时移/偏频/噪声/DC/taper/多径延时及 PA 系数的若干范围也乘 scale。因此下表区分 base 与实际范围，概率多为批级开关，其后幅度/频偏按包抽样；PA 为按包 mask。label 传入但 class-signature=false，不据此证明 PA 保持真实身份。

|操作|base；scale 后执行|主要边界/归因问题|
|---|---|---|
|DC|p.30→.030至.105；单 I/Q 偏置 ±.02s|改变参考及零频；不是 RX 硬件替换|
|taper|p.25→.025至.0875；alpha[.02s,.10s]|作用包边缘，可能经全局处理影响判断|
|循环移位|p.35→.035至.1225；max round(32s)=3至11点，gather modulo256|内部切片接缝和固定参考[80,160)位置会改变|
|增益|p.45→.045至.1575；[.9,1.1]不缩放|native 前级 RMS 在其前；后续某些分支的包归一化可消去纯标量，不能默认全网保留或全消去|
|全局相位|p.45→.045至.1575；[0,2π]|不等于 CFO，不等于真实新 RX|
|CFO|p.35→.035至.1225；±.0004s cycles/sample，25MHz即±1至3.5kHz|旧 ordinary 物理扰动真实启用；NoExplicitCFO 是身份输入边界，不代表旧 CFO 增强关闭|
|相位噪声|p.30→.030至.105；sigma最高.006s rad/step|累积相位过程，不能当纯相位旋转|
|普通多径|p.18→.018至.063；2至4 taps，maxdelay round(4s)=0至1|复 FIR、左 reflect、随机 tap 归一化，可能与 practical 重复损伤|
|AWGN|p.40→.040至.140；20至36dB|基于当时波形功率；若先于 practical 可叠加两次噪声|
|PA|p.14→.014至.049；normal no_pa=false；mpσ.05s、memσ.04s、AMPM.2s、IQimage.02s|追加非线性/记忆/IQ效应未认证标签保持；不能由删除 PA 冻结头改善推断训练必然更好|
|DAC/class-signature|p_dac=0；enable_class_signature=false|该入口实际不启用；不能根据类默认 true/.3 误报|

本轮关闭整个旧 ordinary L 生成，两行共同使用 raw native IQ；原生 U 弱/强视图独立调用路径不受该 L 开关改变。这是新的匹配 raw-slot F 配方，不冒称历史 F ordinary 配方的单算子对照。未来 A_s 从 raw 直接生成，不把通用 AWGN/多径或本 Ag 默认串在 practical 前。

## 唯一新假设与算子

假设：轻微采样落点变化能降低 F 对非身份时序对齐的依赖，改善未参与训练的源 RX 默认头识别。信号处理依据是分数延时的有限 FIR 近似，而不是已有 RFFI 性能结论，参见 [Laakso等，1996，原作者机构记录](https://research.aalto.fi/en/publications/splitting-the-unit-delay-tools-for-fractional-delay-filter-design/)。±1/8 sample 是事前固定的轻度工程范围，不是从目标调出的量或实测 RX 分布；25MHz 下约±5ns。不模拟真实 RX 更换，不保证标签不变。

每 L 包独立 uniform[-.125,.125] sample；33 tap Hann-windowed sinc，有限 reflect，两路 IQ 使用同一实系数，移除16点数值延迟，不用 roll，不加 CFO/PA/DAC/噪声/剪裁。变换后恢复输入包 RMS。内部[16,240)在恢复 RMS 前对 reflect/edge 精确相同，原参考[80,160)的支持在原包内；端点仍能经包 RMS 或网络全局运算影响结果，不宣称边界无影响。有限 FIR 有频率近似误差；本地单频与脉冲检查只证明实现，不认证全部 TX 身份保留。

## 最小 E20 源实验与放弃规则

F保持三路 SIG_TIME、统一 Sinc 物理更新、900722总参数、真实joint160、默认cosine6×160/scale30，正常联合训练。每模型从零20轮=13L+7U/4440更新，无任何父权重/teacher继承，不冻结或另训头。RX1在全部训练、U、V、probe/calibration中排除；有效 L_train5040、hidden U45360、seenV21600。held原L1260只在final20开发评价；144 fit anchors由既有fold materialize读取但不用于训练或选择，成本披露。该契约不同于 fullL6300 或5580/720。

令 RD(v)=同一 v 两次随机前向的平均 smoothedCE+.01标签平滑以及alpha1对称KL。

- B00：.5RD(x)+.25RD(x)+.25RD(x)。
- B10：.5RD(x)+.25RD(Ag(x))+.25RD(x)。

两行均6次完整L前向，总目标尺度1，额外 optimizer/教师/部署参数0；无跨视图KL、无As、无新U目标。g只生成一次，各对不重抽。延时使用私有 CPU generator seed20261016071+step，不改变 loader/dropout/U RNG。所有合法 L 标签保留，U TX始终隐藏。实际每行3,409,920次L前向，相对历史F的2次L前向为3倍；U/EMA/V/smoke/开发评价另计。

主候选只有B10。完整两个E20且final L CE≤.5后，heldaccuracy至少+1pp、macroF1不下降、seenV均值下降≤.5pp才保留假设。否则结束此固定候选，不换强度/RX/seed救援。源开发数据以前可能被其他历史模型见过，不宣称新的独立盲测或跨seed证据。

本轮是明确预登记的source-only最小证伪；不生成目标clean/practical预测。若以后源证据支持，另登记full-source B00/B10/B01/B11，固定同一三槽与同预算，As直接取原practical，全部合法有标签satellite监督，不附困难学生高置信条件；组合As(Ag(x))延后独立检验。正式row须自动冻结、clean+六practical全部prediction固定，再独立truth-last。每个suburban与自身clean差≤2pp先作探索目标，同时要求clean相对匹配B00正收益；不能以降低clean通过差距目标。

## 验证、发布和状态

本地有限算子/目标/5个负例检查通过；两个真实F的合成联合更新、同初始化、实际6前向/对内相同IQ、默认头更新、joint160导出/去W边界和4个权限负例通过。包含1536个训练用合成包前向，额外边界读出16；没有真实数据/目标读取。已有本地CPU PyTorch2.1对NumPy2的bridge警告保留，检查没有使用NumPy桥，也没有安装/修改环境。

普通N607现有环境preflight通过；精确CWD/命令、不可变root、唯一owner、跨owner每GPU最多3总进程和12000MiB空闲要求见登记。实际发布前仅一次聚焦P0/P1审查，release单归档比对/remotecompile；launcher首先每row自有真实scratch checkpoint no-query smoke，PASS立即训练。失败保留产物，停止新pending，健康任务继续，不自动重发。

REJECTED_EXTRA_GATE：通用设计批准、固定规划文件、全库测试和额外审核层不属于用户允许的阻塞项。本轮执行必要输入权限、真实代码commit、聚焦验证、一次P0/P1、最小登记及发布读回，不另增审批。

当前唯一待证实结论是B10是否改善held源身份识别；联合clean正收益、low urban>60%、全部suburban接近clean与跨seed复现均未完成。
