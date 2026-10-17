# F＋跨 RX：STF 窄频带证据可用性检验

当前LOCAL_IMPLEMENTED／未启动真实源IQ或训练。run_id=20261017-phase1-FcrossRX-bandavailability-sourceRX1-e20-s2026092701-m4-r01。固定F＋跨RX同TX、三SIG_TIME/统一Sinc/joint160/NoExplicitCFO/defaultcos30正常联合训练，从零E20/model。已有量化格点、时序、session、相位/频谱bank等固定负结果保持关闭。

唯一假设：不改变原公共raw绝对监督份额，轻度改变一对已知STF谐波的可用性，能否使现有CVS学到更分散、可跨RX迁移的TX身份证据。这个假设待证；有限频带过滤不是替换真实RX，源RX频谱关联不是因果硬件辨识。新算子不拟合或重放旧33tap/DCTbank，不改变其已失败PSD保真标准。实际旧bandedge_taper是时域Tukey窗，不能把名称当频域mask已运行证据。

算子固定：k∈{1,..,6}按私有step/view流均匀、与TX/RX/day/标签/预测无关；512点扩展DFT的±k/20 cycles/sample上对称Gaussian衰减，σ=1/256，峰值depth.5；reflect128后中心裁256，再将整包差分统一缩小到relativeIQ RMS≤.1。无逐点剪裁、丢包、重抽、PA/DAC/CFO、额外噪声、多径、RMS恢复。k对应预先已知20点STF周期，不把梳状谱当RX杂散。相位为扩展DFT正实响应的意义，不能声称裁后FFT256相位逐点不变；反射是人工延拓。sourceguard另用reflect512/FFT1280 DSP参考量化边界，不送模型。

四个新控制均保留原完整公共base .75raw＋.25ordinary/SupCon，再额外.25同IQ两次RDrop；8L/总pairweight1.25：RAW额外raw；GAIN为raw乘以MASK输出RMS/rawRMS；ORDINARY复用第一对已生成完整ordinary，不重抽；MASK为新遮蔽raw。各row都生成相同MASK/GAIN shadows，使生成成本和私有随机流可比。RAW实际总raw系数1，其他组共同raw.75，公共raw绝对份额均保留；不把新增视图偷偷稀释公共raw。原ordinary/13L＋7U/oldgate/ownEMA与原SupCon层位、温度、系数不变；0新loss、NN分支、投影或分类头。原默认头和骨干正常联合更新，部署及export160不加mask模块。

先实际执行一份匹配数据角色CP的2合成包noquery兼容smoke，PASS直接继续两个固定既有seed01/02 B00默认头的源guard，绝不作为新训练初始化。每model只用原fitL5040/72cells各70：raw＋4MASK＋4GAIN，共45360NN；两model90720NN，首parent兼容smoke2另列，3CP读取。原sourcecache整体可反序列化L6300/未用Vfeatures，但先按合法effective IDs排除heldRX1后只消费fit5040，U IQ/真值和V/held/target不用于guard。两模型各完整四draw和原正确集合、RX/TX/day全部保留，0拟合诊断头或更新。

guard固定通过条件：MASK pooled及每draw相对raw accuracy/F1≥−.5pp、raw正确保留≥99%，每RX/TX/day accuracy≥−1pp；IQ变化median≥.01/max≤.100001、matchedpower相对误差≤1e−6、长DSP参考误差能量/变换变化能量≤.01。两model全部满足才自动运行四新scratchE20；任一失败仍完成两guard后关闭固定频带/宽度/深度/cap方案，0新E20，不换RX/seed或改阈值救援。guard只是sourcefit身份保护，不证明Ag有效。

条件训练阶段原L5040/U45360/V21600、held源RX1原L1260 FINAL20；split392005、model/globalordinaryseed2026092701，maskseed202610171951＋step；sourceeval原20261016103，训练/源评价与正式测试流分离。每组E20/4440、4546560 LNN含1136640extra，568320生成MASK/GAIN shadows；记录432 TX×RX×day×k cells及5040L×6pair覆盖、原全部L/U顺序、每步loss/grad/20轮、实际IQ/RMS/cap/门控/默认头联合更新。每row完成立即自动冻结原defaulthead并读出源clean＋六practical，使用原物理配置和流；不等待整矩阵才测，不连接正式target。

固定primaryMASK，对RAW/GAIN/ORDINARY各须heldclean≥+1pp、cleanF1不降、seenV下降≤.5pp及完整健康预算/最终base目标≤.5。六practical全部报告，独立Agclean资格不要求它们已改善；以后仍须独立seed及完整Ag/As四格交互，不能由clean涨分推导联合成功。任一独立资格失败关闭本固定算子，不调width/depth/k/RMSbound/weight/RX/seed/epoch/阈值救跑，不将控制事后晋级。

预算最大四模型80modelE/17760更新/23293272行内NN，加sourceguard90720和兼容2，共23383994；若guard失败则新E20为0，仅90722NN。两旧CP父成本40modelE/8880/9373356NN单列，0新模型继承。额外FFT/描述/生成有计算成本；本地已执行134合成IQ primary/longDSP、5算子负例；真实CVS四组4120合成NN/4optimizer更新/206同初态/900722参数/共同loss和shadows精确一致；4配置和10科学/guard反例PASS。已知NumPy2桥警告保留，未调用桥、不安装环境；所有本地真实IQ/CP/target均0。唯一限定P0/P1正在进行，不新增签名/receipt/hash链或设计审批。

一run/owner、不可覆盖root；CPU2 sourceguard、共享跨ownerGPU锁计入preCUDA与实际NVIDIA进程，≤3/GPU，健康旧任务不热改/停/重启。一次普通账户预检/一个release archive单次SHA比较/compile；实际PID/CWD/argv/GPU/日志增长核验。权限/来源/非有限/预算/输出碰撞等技术失败保留产物，不自动重跑，其他健康worker继续。完整矩阵/commands/CWD/paths/stop与预期产物以experiment.json为唯一维护位置。

完整目标仍为新Ag的跨域身份价值、As适应和同模型联合收益；当前无新heldsource性能、更无正式clean/六practical、low>60%、suburban接近自身clean或跨seed训练证明。不会用源guard、梯度健康或本地测试替代这些证据。

独立 P0/P1 静态审查：PASS；未发现会直接导致错误运行、越权、破坏输出或无法合法预测的缺陷。未读取真实 IQ/checkpoint，未运行网络或远端任务。N607 单次只读预检：PASS；尚未发布或启动本 run。

技术运行 r01：FAILED。父模型真实 checkpoint 兼容性 smoke PASS（1 次 CP 读取、2 个合成 NN 前向、0 个源 IQ）。独立 guard 子进程缺顶层 experiments 包而退出；0 guard IQ/NN，0 E20。原 PID/子 PID 自然退出，输出保留。隔离 Python -I 已复现并验证最小 namespace 修复；r02 仅技术恢复，全部科学参数/预算/阈值不变。

技术恢复预登记：20261017-phase1-FcrossRX-bandavailability-sourceRX1-e20-s2026092701-m4-r02；仅修复独立包导入，科学计划保持。

技术修复验证：第一次 parent namespace stub 不覆盖原后端依赖，定向复审发现同一启动 P1；最终撤销 stub，先以 stdlib spec 读 guard_design，再完整初始化旧后端，再注入本包。两 seed 在 N607 各自 Python -I 完整 bootstrap_guard＋checkpoint/original design 导入 PASS（0 IQ/CP/NN），定向复审确认闭合。原模型/算子检查沿用未变的 r01 本地结果，未重复训练。原 N607 只读预检在本次恢复继续适用，无资源/环境改动。

源域身份保护已完成并独立复核全部 90,720 个默认头预测：两个冻结模型均 PASS。seed01 原始 99.8413%、MASK 99.6974%（−0.14385 个百分点）、GAIN 99.8413%；seed02 原始 99.8413%、MASK 99.6627%（−0.17857 个百分点）、GAIN 99.8165%。MASK 对各自原始正确集合的保持率为 99.8112% 和 99.8062%。这均为 fit L5040/RX3、4、6、8，非 held RX1、正式目标或新训练成绩。

IQ 相对 RMS 中位数 0.100000，97.3909% 的视图触发预定上限；最大 0.1000000047。反射边界误差/变换能量为 1.56265e−7。两模型逐样本变换随机流完全一致。保护计算各耗时 31.0836 s、30.6381 s；独立完整预测复核 4.5079 s，0 新 IQ/CP/NN。

两个 guard PASS 后四个原预登记 scratch E20 row 已自动启动。独立核实 RAW/GAIN/ORDINARY/MASK 的 PID、start ticks、CWD、cmdline、CUDA 卡 1/2/3/4 与输出一致；实际 leo_practical 导入来自原固定 release 的 experiments/adv3b02_xuc/code，residual/post_sync/EQ=false（G=1）/reference_snr_db=30/quality adaptation+IQ compensation+phase tracking=true/reflect 均保持。全部四组无 As 训练，原 U/own EMA 继续；当前 RUNNING，未得出 Ag、urban 或联合模型有效结论。

离线全量复核器：真实 90,720 预测核对 PASS；4 个协议负例 PASS。四组在第 2 epoch 的更新文件均增长，训练尚未完成。
