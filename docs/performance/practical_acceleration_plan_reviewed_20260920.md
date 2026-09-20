# DAOT＋FastTrust-RC4／Practical LEO训练加速方案（二次复核）

日期：2026-09-20。本次是中断后的整理、代码复审和方案交付，没有启动新训练、修改活动进程或变更训练配方。

## 1. 先对齐已完成与未完成

上轮执行优化提交为`45bda90a3dd63f7f2113b9cdad41dfd6488fa2d7`，位于`codex/daot-practical-three-20260918`。包括未消费源域星地验证分支修复、固定评估IQ缓存、未返回GPU元数据的构造跳过、同batch固定Receiver复用，以及fast路径无裁剪时的梯度快照复用。上轮记录25项相关测试通过；本次未改运行代码，不重复执行这些测试。

本次重新读取了实际训练、Practical适配器、信道、模型、验证、周期预测/评分和日志写入路径，补充了缓存位置、六场景接入、同步评分、历史日志重写和配置常量重复计算等问题。之前的26项审查仍保留在[第一次审查](practical_optimization_audit_20260920.md)。本次代码定位与配置摘录见[复核证据](practical_acceleration_review_evidence_20260920.json)。静态确认存在计算，并不等于已经测量其耗时占比。

服务器当前进度未重新查询；耗时来自2026-09-20已保存的完整日志审计快照，不能据此声称各进程此刻仍在运行。N607现有release未在本次更新。已实现的缓存默认关闭，新增命令参数本身不能让旧release获得优化。

## 2. 不变的训练与评估条件

保留DAOT＋FastTrust-RC4、200epoch、L/U步数与样本角色、clean＋星地真实拼接前向、EMA更新规则、现有损失和采样随机性。保留E100、110、…、200的11次测试；测试仍在独立预测完成后由独立scorer连接truth，分数不进入训练控制。

四条信道配置分别是full不均衡、full正则化及增益限制ZF、full MMSE、residual不均衡。ZF的正则化和增益限制不能为加速取消。residual是独立的等效残差近似路线，不能把full偷偷换成residual并称为同方法加速。

原LEO与LEO_WEAK继续保持独立含义，不与Practical混用。原始数据、历史checkpoint、日志和已完成测试保留原路径。

## 3. 现有耗时结构：先优化主体

已存日志快照覆盖full_noeq E1–142、ZF E1–129、MMSE E1–152、residual E1–200及原LEO E1–200。四个Practical组的训练批次约占已记录总耗时的74%–81%；其余主要是源域验证和周期目标测试。因此只加速测试，无法解决大部分训练时间。

旧周期测试为168000条IQ×4场景，单次672000条预测，其中504000条需要星地增强；历史单次中位耗时full约39–48.5分钟、residual约22.5分钟。这些是共享资源下的快照统计，不是隔离硬件benchmark。

现有路径为：

```text
CPU读取IQ → GPU常规增强 → GPU到CPU float64
→ Python列表 → NumPy → 逐记录Practical模拟
→ Python列表 → GPU张量 → 教师/学生前向与反向
```

固定评估虽不需要GPU常规增强，predictor仍先把clean IQ送上GPU，再调用以上CPU信道路径，形成可避免的往返。

## 4. 二次审查结论与处置

|项|代码证据与判断|优化决定／状态|
|---|---|---|
|源域几何验证未消费星地输出|四组多视图几何关闭，旧函数仍生成并前向星地IQ，最终只使用clean特征|已修复。每轮V=27000，200轮约540万条无效增强与对应前向；不是三个场景各做27000条|
|固定评估反复生成相同IQ|实际输入、ID、完整信道配置和随机实现一致时，跨checkpoint可复用|已实现按batch磁盘缓存；本地12case合成CPU视图获取快10.2–31.4倍，未证明整轮同倍率|
|缓存位于GPU回传之后|`apply_practical`先执行CPU float64及tolist转换，再查缓存；命中后仍以tolist构造GPU张量|新增高优先级：固定评估在CPU端完成生成／读取，再只上传最终IQ；不能只加第二层缓存而保留原往返|
|GPU元数据无消费者|`return_meta=False`时无需构造states/SNR/CFO等返回张量|已修复；不等于Python元数据计算全部被移除|
|信道完整Python元数据仍每条生成|full/residual都执行配置哈希、字典展开和input/output waveform metrics；cache helper仍请求完整metadata|配置不变量可复用；其余需追踪诊断消费者后提供轻量／完整两种元数据路径，默认保留必要质量、锁定、均衡、ID证据|
|配置常量逐记录重算|`Config`不可变；`config_hash`每次asdict＋JSON＋SHA；ChannelStream每条重复计算diffuse、平均LOS功率、PDP、各状态delay等|缓存只依赖完整Config的只读常量；随机geometry、state、ray_coef、相位、噪声及stream历史仍逐记录独立|
|同session Receiver重复初始化|硬件参数仅依赖配置、receiver_seed及session|已实现同batch复用；跨batch受限LRU可作为后续小优化，禁止共享有状态ChannelStream|
|梯度反复扫描与同步|常规路径finite逐参数item，再做裁剪前总范数、裁剪后总范数及多个子模块范数|高优先级：合并观测、批量读回；异常守卫逐step保留。现有fast路径可复用，但不能无验证全局开启|
|fast总开关捆绑多项变更|`a1_runtime_fast`同时影响梯度、RC4教师identity-only和部分统计读回|把收益按子项定位，分别验证；最终是否启用总开关由同环境保真检查决定，不能把它当单纯日志开关|
|RC4教师计算完整双骨干|两个弱视图已拼接一次前向；fast关闭时仍走完整模型|优先验证只执行identity分支；不要再次声称“合并两个弱视图”是新增优化|
|目标predictor只消费tx_logits却调用完整模型|`return_aux=True`进入完整双骨干路径，select_identity_logits只取tx_logits|高优先级只读优化候选；验证logits、预测、缓冲区、输入参数与可选适配器一致后使用identity-only|
|clean源验证和几何验证两次前向|同轮先evaluate_loader，再tail_geometry取z_id；两者对domain_labels的传法不同|收集一次符合两者条件的输出，复用z_id/labels/domain；必须先证明domain参数、顺序及状态等价，不能直接拼接替换|
|合并验证可能改变RNG|少迭代一次DataLoader可能减少iterator/worker种子消耗|验证全局及独立generator、下一轮采样顺序；用独立评估generator或保留既有推进量，不能只比较本轮accuracy|
|最后一轮重复几何检查|训练结束再次调用tail_geometry，可能与末轮结果相同|仅模型版本、输入、参数、回滚状态均相同才复用；不能仅凭epoch=200复用|
|每轮重写全部历史日志|`_write_ssdg_epoch_telemetry`以w模式重写CSV与JSONL全部rows|新增明确重复：无其他调用时200轮每种格式累计写20100行而非200行。JSONL追加；CSV固定schema或仅字段变化时重建。处理恢复去重和半行，不承诺它是主瓶颈|
|目标预测构造大量字典与美化JSON|每条包含重复run_id/row_id，最后整体indent=2序列化|低于信道优先级；可流式写同一合法JSON格式到临时文件，完成后原子发布。改紧凑schema则需predictor/scorer一起迁移|
|周期测试同步等待CPU评分|`evaluate_checkpoint`直接运行predictor，然后subprocess.run等scorer，完成才进入下一epoch|先保持GPU预测同步，把CPU评分排到独立任务；训练只接收覆盖/失败状态，不接收分数。最后必须等待所有评分结束再标完整完成|
|测试重建模型及加载checkpoint|当前在同PID中重建独立模型，这是保护训练状态的方式|先优化信道和identity-only；模型常驻复用列次优先，加载每个冻结checkpoint并重置状态，不能直接把训练模型拿去评分|
|checkpoint payload每轮构造|state_dict本身不等于完整GPU复制；周期保存、最终导出等有实际消费者|不要把每次state_dict都报告为磁盘写入；先量化，再延迟昂贵序列化／深拷贝，并保留必要异常checkpoint|
|num_workers=0|信道在取batch并上传GPU之后运行|仅增加loader workers不能并行主线程信道；需要独立信道生产队列|
|CPU逐记录及逐时间点循环|full含跟踪/多径，residual仍初始化同源latent并做跟踪递推|保真分块并行和编译热点优先；不能为加速改变跟踪方程、噪声或均衡策略|
|NumPy兼容桥|既有N607探针显示from_numpy失败而as_tensor(arr.copy())成功；adapter仍保留列表路线|选定一个经子进程探针验证的数组／缓冲区桥接路径，失败回退；不要每batch试错，也不要热升级活动环境|

源域`eval_sat_channel=false`，所以当前不会额外运行完整的三个源域星地准确率测试。上轮删除无效几何星地分支后，固定评估缓存的直接主要收益在目标测试；不能把源域缓存和删除分支的收益重复相加。

## 5. 不能以“重复”为由删除的部分

- L按U长度循环重采样属于222步预算；删除会改变样本曝光与优化步数。
- clean_duplicate或相同IQ仍可能产生不同dropout、BN统计及梯度；拼接结构要保持。
- E80之前即使卫星CE权重尚未启用，拼接卫星样本仍参与共享BN前向。
- 拼接增强、DAOT的high视图不能按场景名合并，输入常规变换、role/view和seed可能不同。
- EMA教师输出不能跨权重更新缓存；仅同step、同输入、同teacher版本可以复用。
- Fishr、orth、RC4守卫和有效损失是方法组成部分；降低困难场景比例、减少视图、减少epoch或测试次数均不是纯执行优化。
- `zid_leakage_probe_required=false`表示不作为必需阻断条件，不表示诊断毫无价值；执行开关与阻断开关要分开。

## 6. 六环境需要补齐完整入口

此前“各个环境都需要”应覆盖：`practical_high`、`practical_mid`、`practical_low_urban`、`practical_low_suburban`、`practical_mid_urban`、`practical_high_urban`。

复核确认：独立`E:/type10-7/code/leo_practical/channel.py`已有六环境，当前训练工作树内channel和adapter.PRACTICAL仍只有前三个；predictor与scorer共用的validate_scenarios只接受固定clean＋三场景元组；周期测试expected_records还默认672000。因此不是增加配置字符串就能完成六场景接入。

下一版应统一场景注册表，并贯通信道、适配器、训练计划、预测、评分、覆盖检查及缓存。评估预期条数从实际无标签manifest样本数和冻结场景列表计算，同时检查ID唯一性、每scene完整覆盖，不能取消覆盖检查或从truth推断数量。保持旧三场景顺序/种子映射，新增场景使用明确稳定映射。

四种配置×六环境至少覆盖24个信道组合。旧25项测试只覆盖四配置×三环境，不能称六环境已验证。六环境训练的混合比例尚未在当前配置中定义：基础设施全部支持，训练计划作为新配方显式登记；本方案不把旧三场景概率静默平均拆成六份。

按同一168000条评估输入测算：clean＋六环境为每checkpoint1176000条预测，是原来的1.75倍；E100–200共11次、四组共51744000条预测。这是计划工作量，不是已经执行的结果。

固定IQ仅float32信号部分：四组×六环境×168000×2×256×4bytes约7.69GiB；共享clean约0.32GiB。元数据、索引、文件开销和生成暂存另计。按实际样本测量最终磁盘预算，不把8GiB当完整缓存容量。

## 7. 推荐的最终执行结构

### 固定验证／测试：CPU端物化一次，模型每次重新推理

```text
无标签CPU IQ及物理ID
    ↓ 完整配置、处理版本、scene、seed绑定
固定Practical IQ分片（首次生成，后续读取）
    ↓ CPU连续缓冲区／按需pinned staging
单次H2D → identity-only预测 → 临时预测文件 → 完整性检查＋原子发布
    ↓
独立CPU scorer → 每个checkpoint独立score与完成状态
```

固定IQ可以跨checkpoint复用，预测/logits不能跨checkpoint复用。源域多视图验证只有实际启用且输入/seed固定时才物化；不要为关闭的分支预生成无用数据。当前NPZ缓存可先用，规模扩大后考虑连续分片或mmap以降低小文件与JSON解码开销。

新存储不能只凭ID命中：绑定实际预处理输入、配置和实现版本，并在输入改变时失效。一次性转换旧键后才允许CPU加载器按已验证的固定包读取，避免每次重新上传GPU再哈希输入。

### 动态训练：有限预取队列，按原视图计划生产

```text
batch／视图计划（epoch、step、role、view、IDs、scene、实际seed）
    ↓ 保留常规增强→Practical顺序
CPU信道worker分块生产 → 有限队列 → GPU消费
        生产下一批             前向／反向当前批
```

第一步可先在同batch内并行独立record，不改变训练调用顺序。随后再重叠未来batch：常规增强当前在GPU，不能直接把原始IQ送入CPU信道来“绕过”它。只有完成RNG依赖拆分与对照，才能提前生成下一批常规增强；GPU dropout、loader、增强和信道generator的推进不得因预取深度改变。

初始用小型有界队列并测试少量worker，记录CPU/内存带宽及线程超配；不固定承诺worker越多越快。task携带完整IDs和种子，不能依赖父进程ContextVar自动传到worker；按step/view有序回收，保持末尾小batch正确。取消／恢复时复原消费位置和RNG，不能把未消费预取结果当已训练数据。

长期可把递推、分数延迟滤波等热点编译或批量化。先保留参考float64／complex128、逐record独立随机流及数值顺序；更改精度、随机算法或规约顺序时，必须说明由严格等价转为有界数值等价，不能宣称逐位一致。

固定一个训练增强版本反复使用会损失多样性。完整按epoch／view计划预生成可以保留预定多样性，但存储量大且依赖正确的前置增强；滚动预取更合适。只缓存信道参数也不能跳过依赖输入质量的接收锁定、均衡及AGC计算。

## 8. 分阶段落地及验收

|阶段|交付内容|验收条件|
|---|---|---|
|0：基线与profile|独立release，原始配置与上轮已修复版本的同源输入对照；记录资源竞争|不改活动进程；分出读取、普通增强、D2H、列表/数组转换、初始化、信道、H2D、教师、学生、反向、诊断、验证和写盘耗时|
|1：确定冗余|使用已完成修复；梯度统一观测；配置常量复用；JSONL增量写入|相同输入/seed下IQ、loss、梯度、optimizer/EMA、BN、RNG及下步顺序相符；异常守卫正常|
|2：评估数据路径|CPU固定视图物化、identity-only预测、一次clean验证输出复用、独立CPU评分重叠|预测记录逐ID一致；当前batch划分和末尾小batch正确；变化的loader/RNG推进已处理；预测全部完成后才评分|
|3：六环境接入|注册表、适配器、场景计划、预测与scorer覆盖统一|24组合＋clean，均衡参数和seed隔离，新增环境不改变旧三环境输出；新六场景配方单独登记|
|4：训练流水线|先同batch分块并行，再CPU/GPU重叠|队列深度／worker改变不改变计划与训练消费次序；错误可定位；CPU线程不超配；真实GPU等待下降|
|5：热点内核|编译递推／滤波与批量计算|逐项记录严格／容差等价边界；全链路吞吐优于参考，不能只报单函数倍数|

阶段0需分别选取模块激活前后和课程阶段的源域固定短窗口，例如E1–20、E21–40、E41–79、E80–90、E91以后；不必为profiling重新训练全部200轮。使用合法源域状态，合成输入则明确只是算子证据。记录GPU型号、共驻任务、线程数、batch、AMP、Torch/NumPy版本、冷／热缓存及每项启用状态。

GPU阶段用CUDA event或timeline核实异步执行，CPU用单调时钟；只在有界测量窗口边界同步，不在每个record或每个算子强制synchronize，把测量本身变成瓶颈。报告稳定窗口中位数及尾部耗时、每秒处理样本数、GPU等待和CPU队列等待。先单项开关，再组合对照，避免把重叠收益简单相加。

`T_step`目前接近串行各段之和；流水线理想下接近`max(T_CPU生产,T_GPU计算)+不可隐藏的搬运/同步`。整轮还需加验证、写盘和周期测试。若某阶段占比p、局部加速k，理论整体上界由`1/((1-p)+p/k)`估算；例如仅假设p=0.6、k=2，总体约1.43倍，不能写成2倍实测。

## 9. 本次交付的准确结论

优先方案是：已确认无用分支修复＋固定评估在CPU端读取＋只读identity前向＋梯度同步合并，然后做同batch信道并行和受控预取。历史日志重写、元数据和配置常量属于确定或可进一步确认的小开销，按profile安排，不抢占主瓶颈工作。

本次新增的是复核报告、定位证据和实施／验收方案；没有宣称六场景已接入、异步流水线已完成、N607已获得加速，也没有把上轮局部缓存测速外推成整轮倍数。正式部署前仍需同环境源域保真及吞吐验证，后续状态以新release和实测证据为准。
