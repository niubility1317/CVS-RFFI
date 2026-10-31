# Phase3独立时频恢复器：源配对、真实F0与成本

VERIFIED / ANALYZED_SOURCE_COST_ONLY。两种目标都达到事前成本条件：中位步骤耗时至少减少10%，三个块均更快。这里没有优化器更新或新恢复性能结果，不证明同质量完整训练提速。

|目标|DURA参考中位步骤ms|新TF中位步骤ms|耗时减少%|三个块均更快|
|---|---:|---:|---:|---:|
|IQ＋谱|128.642678|31.555337|75.470553|3/3|
|IQ＋谱＋冻结F0指导|172.115650|69.454961|59.646342|3/3|

![同卡源步骤成本](E:/type10-7/automation_reports/CV-SincNet/20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r02/source_cost.png)

单GPU4、B256 resident输入，同源样本交替执行。每目标/模型3块、每块2预热＋6计时，合计24预热、72正式计时、96次forward/objective/backward/clip1。计时含纯IQNMSE＋0.05三尺度STFT幅度相对L1；指导版另含0.1×实际F0 fused160余弦误差和逐条输入梯度计算，梯度cap为IQ梯度范数的0.5。目标缓存和cap权重detach，F0/default head不更新。

## 已完成的实际Phase3接口

原Tri-SIG-TIME SSL F0为20261027-phase1-pamf-repl-shuffled-views-m16-r02的main_split-repl-shuffled_time-3view-s2026092701，执行3ff8f4b7e85b5608f5b45350aa18081499a7ef1d，232993参数、固定E50/11100及其真实默认六类头。严格loader核对原source392005/6300L_s/56700U_s/27000V、原source-only轨迹和三视图固定时间重排。不是此前旧F50。

新TF完整196898参数、19646464 Conv/Linear MAC，六个全率时/频域块；六rFFT与六irFFT各64通道，768个实通道变换，0CG。FFT/GN/激活不能用MAC省略。新G从随机恒等输出开始，没有DURA权重或G祖先。成本参考是原训练好R69的独立深拷贝，eval状态下开放参数梯度，但没有更新；它不是新G初始化或IQ教师。两个初态不同，不能把速度比称为同质量训练速度比。

真实2条L_s smoke已PASS，实际F0输入VJP与cap、新TF输出头梯度、原默认六类头、原DURA native与生产graph路径一致、全部模型参数/buffers不变。smoke 1.803007秒。成本步骤末清空梯度，F0及原DURA没有参数grad，所有四个模型state前后相同，0optimizer.step。

## 原源数据和一次归一化

原角色按位置等距取128L_s＋384U_s，512个独立物理记录、一个既有G10 MIX01视图。L来自d319源缓存、U_s来自已验证Us capsule/split；只打开原P0 received和IDs，不读restored、teacher_logits、L_y或U_s标签。源OE/V/novel/目标U/Q均不作为方法输入。

可信离线定位器读取原2.359GB ManySig PKL整文件，payload包含其他角色及源TX坐标，不能声称没有读入其他角色字节；只索引所选512个源packet值，定位坐标不输出给G。原FP64包先转FP32后构造复数，第一次input RMS形成配对clean。前两条以同physical ID/receiver/day/原MIX01重放，逐值等于原P0，最大差0；其余只从原raw构造clean参考。新raw RMS512次、canonical observe2次、post-channel RMS0，未把旧normalized缓存当raw。独立复算clean能量与1最大差9.21328491277e-09。

原source整体训练L/U为10/90，此成本子集是25/75；只作计算成本，不以此代表未来训练曝光。重复计时不增加独立物理样本数。

## 完整成本和证据边界

新F0 clean特征缓存512×160/327680字节，诊断内存缓存生成0.051585秒，不能与未来63000训练缓存混称。F0原50轮11100步源有标签知识继承，指导版为F-guided，不能称F-free。F0 feature行12806＝smoke6＋cache512＋guided12288，默认头4行；TF和DURA参考各12290行＝smoke2＋48×256，DURA生产逻辑输入2行，内部graph启动预热不冒充两次完整神经调用。

R02计时区段10.419957秒，模型加载1.250774秒，源配对准备（包含smoke）3.290913秒，worker显式计时区段15.327186秒（起点在导入之后）。artifact mtime观测从launch写入到controller终态17.552998秒，包含控制器/解释器等待，不是精确Popen计时。计时步骤不含optimizer.step、IO、传输、缓存生产、加载、正式评估，不外推整轮或完整30轮。

显卡NVIDIA GeForce RTX 3090，共同驻留系统峰值allocated 1518903296字节，不能作为单TF训练显存。controller 2124791、worker 2124809，实际子进程header的PID/CWD/argv/CUDA与controller提交一致，现均自然结束，日志含启动→真实PASS→完整结果。任务约15秒，未捕获live worker /proc或nvidia PID快照；不伪造live占用或为遥测重跑。GPU实际执行由CUDA event、device name和分配峰值产物核实。启动准入前GPU4已有1个compute、剩余23189MiB，新增1个；其他owner不改。

## 技术失败、独立核验与下一步

r01在首个真实smoke错误地把DURA公共单输出解包为两项。47c InferenceEngine先内部检查valid再return result[:-1]，没有reader时只返回(restored_IQ,)；r02仅取[0]和隔离新root，数据/模型/损失/96步及科学条件全不变。r01失败root和日志保留，其artifact观测墙钟7.832999秒不隐藏；没有96步计时、优化或target。唯一直接问题针对性P0/P1复审PASS。

23项聚焦检查覆盖架构、实际分支/频域伴随、输入与数据角色反例、标签不可读、目标detach/cap、资源和CLI；原一次P0/P1 PASS。独立NumPy复算96条完整时序/交替顺序/24预热72计时、全部中位数与三块比例、512源ID/角色/clean RMS和所有计数，最大指标差0，新增G/F前向和truth0，耗时0.049091秒。归档SHA、remote compile和产物独立读回通过，执行b19c2ef6d9b75c5afecbafb4ecd3d44716e73c31。

下一步是同新架构从零单阶段30 epoch的纯IQ与实际F0指导对照，匹配源输入、初始化、预算和完整测试；source OE继续只冻结诊断。源成本PASS不替代六星地逐个距raw clean≤3pp、clean保持、实际Phase3身份/未知开放指标与完整成本验收。默认DURA保持，完整目标ACTIVE。
