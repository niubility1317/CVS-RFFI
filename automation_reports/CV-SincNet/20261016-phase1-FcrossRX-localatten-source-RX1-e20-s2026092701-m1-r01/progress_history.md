# 局部观察衰减：源域预登记

REGISTERED_NOT_RUN。精确row/command/environment/CWD/paths/GPU/owner/stop/artifacts见experiment.json；实际codecommit将在release记录。

# 固定 F＋跨 RX：有界局部观察衰减的 E20 源检验

未完成联合目标。增强前底座为用户最新指定 F＋跨 RX 同 TX SupCon，scratch E20、原 joint160、默认 cosine30 头正常联合训练、NoExplicitCFO 和必要逐包补偿保持。上一轮重复差相关 G0 已因固定质量条件失败关闭，不用该残差注入噪声；当前没有通过源规则的独立 IQ Ag。

唯一假设是：在不改变采样位置的条件下，有界削弱短片段能减少对局部观测的依赖，促进当前模型利用仍可靠的身份证据，最终改善跨 RX clean 并保护 practical。这是待检验的 IQ 观察正则，不是纯接收噪声、RX 硬件替换、真实 LEO 快衰落或已证明的标签保持操作。已有相位/观察原点失败不能支持该假设为真；本轮也不通过改变其范围重开旧算子。

固定算子在实际旧 ordinary 输出 x 上生成一个16点窗。起点独立均匀取0至240，无循环连接/裁切/补齐；窗内 g[k]=sin²(π(k+0.5)/16)，其余为0。深度 a=min(0.5,0.1(1−2e−6)||x||/||g x||)，零加权能量取0.5。输出 (1−a g)x，I/Q 用同一正实系数。保持全部256个位置和窗外240个值，无新RMS/AGC/噪声/多径/CFO/PA/DAC。每包相对L2变化≤10%，深度≤0.5；零功率包保留、继续CE，不筛样本。16点相当于25MHz下0.64微秒，仅表示数字操作尺度，不冒称卫星传播时间尺度。

真实旧 DataAugmentation 的 p_bandedge_taper 名称对应时域两端 Tukey 窗，不是频带滤波。本轮不改旧处理；新增短窗有随机位置和整包相对范数上限。旧普通增强包含已知非纯物理操作，继续作为共同条件，不预先宣称其身份保持或Ag有效。

最小矩阵：唯一新 LA16-RX1-s2026092701，复用相同合同/预算/默认头 C0 的已冻结结果、完整日志和主序列，不加载其权重。C0 为0.5RD(同ordinary)+0.5RD(同ordinary)；新行0.5RD(原ordinary)+0.5RD(局部衰减同一ordinary)。各四L前向、每步旧ordinary只生成一次，每个RD内IQ不重抽。原 time_down/t2.pool 的L跨RX同TX SupCon只读取第一对已有前向，温度0.2/最大0.1/warm5/ramp10不变。无额外对齐、一致性、诊断头、参数或部署模块。

沿原 E20=前13轮L、后7轮L+U，L监督持续全部20轮；4440更新/222每轮。原U/ownEMA/旧门控/cos/strong_noise_std=.015不变，U TX真值始终隐藏。只增强L，0As训练。固定fit RX3/4/6/8，L5040/U45360/seenV21600，heldRX1 L/U不训练、heldV5400只FINAL20读取。不是全源L6300，也不是内部5580/720。初始化 scratch，无父模型/teacher/EMA/optimizer继承，不加载E50/E200。

LA16位置私有流 seed20261017111+ordered optimizerstep，独立于TX/RX/损失/置信度/波形质量；深度只由该IQ全包与局部能量限制，不从标签或heldV选择。每epoch/step刷新，原loader/dropout/U流保持。记录全部568320个窗口的物理ID、位置、深度、L2改变量、功率变化；标签只用于覆盖报告。全部TX×fitRX×day×16位置区间和全部241全局起点须观察到，全部5040合法L仍有监督。

固定FINAL源判据须全部满足：相对C0 held clean≥+1pp，clean macroF1≥0，seenV下降≤0.5pp，六个固定practical逐项下降≤0.5pp，完整4440更新/正常默认头与骨干联合更新/无U真值/完整相同主L-U序列和有限预算。FAIL关闭固定局部衰减，不调width/depth/L2bound/probability/ratio/RX/seed/epoch/阈值救援。PASS仅支持另登记源确认，不自动扩目标、多seed或Ag+As。suburban与自身clean差距按≤2pp探索目标报告，不以clean下降缩差。

FINAL逐row自动源clean+原六practical读出并独立复算，源视图不构成正式target测试。正式目标IQ/信道/SNR/困难包不改；真正B00/B10/B01/B11、交互与跨seed正式结果仍待Ag源证据。只有另预登记冻结候选才进入正式truth-last流程，目标分数不回流选择。

成本：新20模型轮/4440更新/2273280 L网络包，原U/EMA/源V与七视图另计，按既有计数预期3579018总NN但以实际为准。旧C0原E20/4440/3579018NN单列，不把结果复用写成免费父训练或权重续训。新增568320次局部IQ构造无新NN；构造与训练/评价墙钟分别记录。预算结束不等于收敛。局部验证1024合成NN/2更新、735直接合成变换包、384采样包和256管线变换包，18负例通过；206初态与C0逐张量一致、原ordinary精确一致、全241起点人工公式、窗外不变/零包保留/范数限制/私有流/原两前向SupCon和原头SIG3-Sinc联合梯度均PASS。0真实IQ/CP/target。现有SSR NumPy ABI警告未阻止上述纯Torch检查，没有安装包。

唯一owner为codex/root/FcrossRX-localatten-source-20261016。普通N607账户、CPU dispatcher2线程，worker动态选择≥12000MiB空闲且所有owner总进程<3的GPU，不改健康任务。不可覆盖run/root、一个release归档单次SHA比较、remotecompile；实际own scratchCP no-query smoke为训练首步，PASS立即继续。只对本run系统技术失败停止pending并保留输出，不自动重试，不触碰外部进程。完整command/environment/CWD/paths/GPU规则/stop/artifacts在同run experiment.json。

REJECTED_EXTRA_GATE：brainstorming技能要求的独立设计审批不在项目最高优先级最小流程白名单；本任务已有研发授权，不增加审批或等待。一次限定独立P0/P1审查只检查会让本次实际运行错误、越权、破坏、不可启动或无法合法预测的缺陷；不做P2阻塞或重复不变审查。

唯一限定独立 P0/P1 审查 PASS：未发现直接阻塞缺陷。只读核查衰减边界/零包保留/同IQ四前向/primary流/默认头与SupCon/发布依赖及全owner保护；0真实IQ/CP/网络/修改。LOCAL_VERIFIED_NOT_RUN，性能未知。

实际执行代码 `3bba9d406fe1dee68b7f76e403f114c11954b862`，GitHub远端分支与本地独立确认一致。一次归档SHA比较/remotecompile/普通账户 preflight 与真实own scratchCP无query smoke通过。唯一worker GPU0，PID/CWD/argv与GPU实际匹配、日志从6516增至11682字节，首轮222步和全5040L/241起点已观察；首轮最大相对L2=0.09999980994737154、平均功率移除=0.02540395709166407，默认头/三SIG/Sinc均正常更新。0As训练/0target，FINAL未知，训练健康不等于收益。

独立读取器已实现：18元数据反例、128模拟窗口记录及原C0全部37800默认头源决策回归PASS，0新IQ/CP/网络/更新。保存的correct计数不一致反例揭示公共helper未核对此字段，已仅在新读取器中补齐；旧代码、训练release及历史结果保持。位置流将独立复放全部568320次；深度/L2/功率变化从保存trace复算聚合，不冒称重新读取IQ验证。该工具不参与训练、选模或参数调整。

观察边界说明：均匀采完整窗口起点不等于每个采样点受到同等衰减，包两端被更少窗口覆盖，后续会从实际位置/深度trace重建并报告该分布。平滑窗仍会改变包络和频谱，不能称为全部特征保持或物理LEO模型；它同时缩放记录中的信号与原有噪声，不是选择性去噪。该说明不改变已冻结的算子或判据。
