# F＋跨 RX 同 TX：轻度接收频偏的输入作用检查

状态：ANALYZED_ENGINEERING_INPUT_ONLY／VERIFIED。真实冻结CP工程检查及独立完整复算已完成，未开展身份实验。前一常数增益源E20已FAIL关闭，clean+0.555556pp但未达+1pp，六practical全降；不调幅度/seed/RX救跑。当前唯一假设：温和接收频偏变化在保留NoExplicitCFO及内部补偿后，是否仍留下值得训练检验的实际默认函数作用。

## 固定最小实验

一个已核实匹配的自有B00/seed2026092701/final20冻结CP，实际scratchE20/4440、无祖先/父权重或target接触，四RX3/4/6/8的L5040/U45360/V21600。复用既有来源核对结论，严格核对元数据/payload并strictload；不读sourceIQ或cache，不读heldRX1/U/V/target，不初始化后续训练。不是完整L6300或L5580/Ldev720训练结果。

一个CPU2线程进程，2合成包realCP no-query smoke，然后32合成包（4family各8，seed20261016421）×5固定视图：identity、正/负频偏端点、对应逆变换。总实际NN162，0真实IQ、0epoch/更新、0teacher/EMA/另训头、0target，无身份标签/准确率。首步smoke PASS立即继续，无额外审批。所有结果写同一不可覆盖root。

IQ算子沿用原生零起点x[n]exp(j2πfn)，n0至255；端点±.00010024013357398965cycles/sample，来自已核实actualordinary base.0004×native E13截断scale.2506003339349741（25MHz下约±2506.003339Hz）。不使用构造默认.0025、E20外推范围或source practical/target挑幅度。无居中、额外常相位/RMS/噪声/multipath/PA/DAC、窗口改写或As。逆变换只作数值对照。

F三路SIG_TIME、统一Sinc、joint160、checkpoint默认cosine头和NoExplicitCFO保持；只冻结推理，不训练/替换头。实际input_policy屏蔽参考响应coord27，ExcitationResponse内部逐包CFO估计/补偿保留，原time/PA身份路径仍接收原IQ；不能把参考分支补偿当作全网不变。观察真实encoder/Sinc输入、time_down/t2.pool/reference、同次forward的joint160/defaultlogits，state和buffers须不变。

## 判断、成本与停止

固定工程条件全部满足才PASS：实际Sinc与生成IQ完全相同、功率最大相对误差≤1e−6；两个逆变换输入相对L2≤1e−6，normalizedjoint最大变化≤1e−4且defaultlogits最大绝对变化≤1e−3；至少一个端点normalizedjoint变化中位数≥1e−3且logits最大变化≥.01。中位数采用32值排序第16个，与Torch lower median一致。所有5×32 joint/logits/power在完成后独立NumPy复算；输入相等、参考屏蔽、冻结state/counter在runtime直接检查，区分证据范围。

任一工程条件FAIL关闭固定假设、0训练，不改频偏/seed/control救跑。PASS只支持另行登记源身份/E20证伪，不是标签保持、A_g泛化或联合目标证明，不自动访问target或训练。以后若有匹配源E20，仍应clean≥+1pp、F1不降、每个6practical和V≥−.5pp，全部同时通过才支持独立确认；本run没有该训练。

输入权限/来源、形状、非有限、预算、不可启动或输出碰撞为技术失败，保留产物、无自动retry，不干预健康任务。工程正负结果都写completion，所有identity/Ag/joint声明为false。无真实source身份评分或正式target测试；联合目标仍未完成。

唯一owner codex/root/F-CROSS_RX-frequency-input-20261016；普通N607账号，CUDA_VISIBLE_DEVICES为空、CPU2线程，不创建GPUworker。精确命令/CWD/环境/路径、六类seed、来源与配置见experiment.json。代码Gitcommit固定；一次N607 preflight、一个archive单次local/remote比对、远端compile；启动后独立PID/CWD/argv/CPU-GPU环境/log增长读回。合成trace/log留私有，代码/登记/小汇总Git交付。

本地10协议负例、4判断反例、canonical原生算子匹配/功率与逆变换/私有syntheticRNG、实际未训练F路径检查PASS；162合成NN/0更新/0真实CP或IQ，单列。一次限定P0/P1审查PASS。当前没有真实冻结CP工程结果，更没有识别收益。


r02仅替代r01标准库同名遮蔽技术失败，旧r01 release/log及FAILED记录保留。修复仅operator.py改名frequency_transform.py/相对导入与直接脚本回归；频偏、seed、162NN/0真实IQ/更新矩阵不变。初始一次审查后，原问题仅一次定点复审PASS。两版本本地未训练F检查各162NN累计324合成NN/0更新，r01远端实际0NN；本r02远端预算仍162NN。


## 实际结果与边界

执行commit `3996730c4d70b36f0e3c4b0a091927dd0a77ffe0`；archive/compile/初始PID-CWD-argv-GPU环境VERIFIED，进程自然退出、completion存在、无failure，独立终态log 417字节。真实CP strictload/no-query smoke及162实际NN完整完成，0真实IQ/更新/teacher/另训头/target。全部5×32 joint/logits/power独立NumPy复算与原结果一致，原固定工程判断PASS。

| 视图 | 单位joint变化lower median | 最大logits绝对变化 | 功率最大相对误差 |
|---|---:|---:|---:|
| identity | 0 | 0 | 0 |
| frequency_positive | 0.0281140576 | 0.800132394 | 1.19209297e-07 |
| frequency_negative | 0.0309339524 | 0.686725855 | 1.19209297e-07 |
| positive_inverse | 2.46831232e-07 | 2.14576721e-06 | 1.19209297e-07 |
| negative_inverse | 2.43528681e-07 | 2.96533108e-06 | 1.19209297e-07 |

输入相等/逆变换input L2/coord27及内部补偿/冻结state与计数为实际runtime直接检查；独立复算范围为完整joint/logits/power数组，未声称独立重算不可见IQ或模型state。工程结果仅证明固定温和频偏对当前完整默认函数有非平凡作用，不能代替身份保护、跨RX净收益或星地收益。

下一唯一科学假设：保留旧ordinary、PA、原U、固定raw监督和默认头的条件下，独立频偏L视图能改善heldRX clean且保护6practical。最小待登记验证是同RX1源折/同seed匹配B00/B10各scratchE20，raw.5＋ordinary.25＋第三槽raw或频偏.25、六L前向；clean≥+1pp、F1不降、每practical及V≥−.5pp、完整健康预算必须全部通过，失败固定配方关闭无幅度/seed/RX救跑。尚未启动该训练；本run全部完成但联合目标未完成。
