# F＋跨 RX 同 TX：轻度接收频偏独立视图 E20 源对照

状态：PLANNED，尚未发布、训练或完成源身份评分。前一固定常数增益候选因clean+0.555556pp未达+1pp、全部practical下降而关闭，不调幅度/seed/RX救跑。当前唯一科学假设：保持旧ordinary/PA、原U、raw固定监督和默认头，独立轻度接收频偏L视图能改善未见RX clean且保护六practical。

## 已有依据与固定矩阵

频偏范围只沿用actualordinary .0004×native E13 cap scale .2506003339349741，即±.00010024013357398965cycles/sample；未按source practical或target挑幅度。单独工程r02已完成真实匹配CP/162合成NN/0真实IQ，正负频偏有实际默认函数作用、逆变换恢复；只支持开展训练检验，不证明标签保持/身份收益。其r01 stdlib命名遮蔽技术失败已保留，实际0NN，不是科学负结果。

| row | raw监督 | 共有ordinary | 第三槽 | 预算与自动读出 |
|---|---|---|---|---|
| B00-RX1-s2026092701 | .5 | .25旧完整ordinary，PA保留 | .25 raw | scratchE20/4440，final20源clean＋6practical |
| B10-RX1-s2026092701 | .5 | 相同.25旧完整ordinary | .25 frequency(raw) | 同预算/前向/数据/初始化规则 |

两row固定fit RX3/4/6/8×day1/2/3×全部6TX，实际L5040/U45360/V21600；heldRX1原L1260仅final20读出，不参与训练/EMA/归一化或选checkpoint。不是完整L6300或L5580/Ldev720。复用已验证carrier/source fold及原角色，U真值隐藏；前13轮仅L，后7轮L＋原U，所有20轮L监督继续。V健康诊断不选模，checkpoint固定最后20轮，0target。

F三路SIG_TIME、统一Sinc、joint160、默认cosine头联合训练、NoExplicitCFO及内部参考逐包补偿保持，无新增部署分支/头/诊断分类器。三对RDrop alpha1/LS.01，每step6个L前向；每对复用同一生成IQ。固定raw.5和oldordinary.25监督，第三槽及对应前向控制显式登记。CROSS_RX仅旧ordinary两前向的time_down/t2.pool，temp.2、maxweight.1、E1至5零/E6至15ramp；raw/frequency/U不新增对比/一致性。

新频偏每包/step私有CPU seed20261016401+step对称uniform；x[n]exp(j2πfn)，n0至255，不居中，不改RMS/窗口/幅度，不加noise/multipath/PA/DAC/CFO身份坐标或As；全部128合法L参与监督，不做高置信过滤。NoExplicitCFO屏蔽coord27，内部参考补偿不删，不将频偏当TX身份或真实RX替换。原ordinary完整抽样/PA/原生min(epoch,13)日程、原U/ownEMA保留。

## 原定判断、成本与停止

全部同时通过才支持独立源确认：B10−B00 heldRX1 clean≥+1pp、clean MacroF1≥0、每个六practical准确率≥−.5pp、seenV≥−.5pp；两row完整E20/4440更新/3409920LNN、3330非零SupCon/13320层位前向项、最终LCE≤.5，B10频偏包观测568320/B00为0，功率最大相对误差≤1e−6、0As训练/头拟合/target。科学失败关闭固定频偏配方，不改频偏/seed/RX/阈值或追加预算救跑。单seed源PASS不是Ag跨域或联合目标完成。

各模型scratchE20，不加载训练父权重；唯一真实CP来源仅dispatcher no-query smoke2，为自有satmix25 B00/seed01/scratch20/4440/final20、无祖先/target，实际四RX角色已核实，strict metadata/payload/load复查；不用于worker初始化/teacher。ownEMA由当前scratch student产生。新run预期总40模型轮/8880更新；每组3409920LNN另实际原U/ownEMA/V/2scratchsmoke/8820held，先前同配方每组4686678NN但以实际计数为准，若一致含公共smoke2为9373358。B10另568320个频偏IQ生成，B00 raw复用，两rowNN视图数匹配。先前工程162NN、其local324NN及本实现两次check4624NN/4合成更新分开记录，不当成真实E20训练。

每row训练后即自动冻结/源clean＋6practical读出，保留全部场景accuracy/F1/CM及TX/day/repair-break/suburban-ownclean差距；全部预测完成后独立复算。hard predictions无logits，运行时meanCE只引用，不冒称独立CE复算。源筛选不构造target，完整源正式四因子/交互项和多seed联合目标仍缺。

唯一owner codex/root/F-CROSS_RX-frequency-E20-20261016，普通N607账号、CPUdispatcher2线程、worker沿用共享容量锁/每GPU最多3个全部owner进程/启动前≥12000MiB空闲，健康任务不改/不停/不重启。精确逐row配置、命令/环境/CWD/路径/六类seed见experiment.json。一个Git实际code/config commit、一个不可覆盖root/一个N607 preflight/一个archive单次SHA比对/远端compile；启动后独立PID/CWD/argv/GPU/log增长读回。

角色/来源/初始化/非有限/实际预算/无法产生合法预测为技术失败，保留产物、停止pending，不干预健康worker、无自动retry；科学低分不早停。首步实际CP no-query smoke2 PASS立即继续，无许可artifact/额外审批。代码/登记/小汇总Git交付，IQ/CP/逐包预测/原日志保留私有。13负例、同初始化、旧B00损失/头梯度精确控制、sixforward/sameIQ/ordinary-onlySupCon/defaulthead jointupdate/export160/privatefrequency/canonical算子与直接脚本回归PASS，一次限定P0/P1审查PASS；尚无实际训练或识别收益。
