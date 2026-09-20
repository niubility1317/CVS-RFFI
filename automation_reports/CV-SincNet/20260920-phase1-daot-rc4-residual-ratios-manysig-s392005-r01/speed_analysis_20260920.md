# 六组residual训练加速审计

截至2026-09-20T21:53:33.732103+08:00，六组进程身份与输出路径读回VERIFIED，仍处于E25—E28。结论：执行优化已经带来阶段性加速，但当前共享GPU争用显著侵蚀实际收益。前25轮相同epoch窗口内，旧residual累计114.75分钟，新六组104.88—111.97分钟，实际加速仅1.025—1.094倍（耗时下降约2.4%—8.6%）。不能将局部信道2.5倍加速或早期约2倍epoch加速写成当前全程2倍。

## 覆盖与限制

完整解析六组160条epoch记录、旧四组652条epoch记录，合计812条；逐行核对JSONL与CSV的epoch序列和耗时一致。完整扫描10份stdout，无Traceback、Error、Warning、OOM/Killed。所有已存在的practical_channel_execution.jsonl完整解析；相关logs.jsonl、metrics.csv、细粒度execution_profile文件未出现。当前配置execution_profile_steps=0，因此只有原生训练/基础验证/重验证/其他四段计时，尚无本次真实训练的信道、IPC、前向、反向及同步细分比例。正在写入的下一轮不计入本快照。

旧四组E161/E153/E138/E200均解析，速度主对照只使用旧residual；full算法工作量不同，不作为residual代码加速对照。六新组改变了拼接场景配方、与旧run不同时间运行且并发负载不同，因此这里是实际运行观察，不是隔离的同输入A/B因果测速。未读取目标测试分数、未调参、未停止或迁移任何进程。

## 六组实际耗时

|组别|GPU|已完成epoch|E2—10中位秒/轮|E1—25累计分钟|相对旧residual累计加速|最新完整轮秒|
|---|---:|---:|---:|---:|---:|---:|
|HIGH_LOWURBAN_75_25|0|26|91.1|104.88|1.094倍|1041.4|
|HIGH_LOWURBAN_50_50|1|27|95.3|111.97|1.025倍|276.0|
|HIGH_LOWURBAN_25_75|2|28|92.8|111.10|1.033倍|267.9|
|MID_LOWURBAN_75_25|3|26|94.4|106.03|1.082倍|1017.1|
|MID_LOWURBAN_50_50|4|28|91.5|110.96|1.034倍|273.2|
|MID_LOWURBAN_25_75|5|25|92.0|109.01|1.053倍|1036.5|

旧residual：E2—10中位192.15秒/轮；E1—25累计114.75分钟；E26—28中位538.01秒/轮。新组E2—10为91.14—95.31秒，对应2.02—2.11倍；新组中当前没有其他GPU计算进程的GPU1/2/4，E24—25分别为273.7/267.4/273.5秒，旧同epoch为553.8秒。这两个较快窗口支持执行优化有效，但不能替代包含尖峰的累计统计。

## 时间为什么突然上升

1. 六组DAOT objective均首次于E21实际启用；E1—20不能代表完整DAOT训练成本。原p=.3/.6/.8日程、E80卫星CE、E100周期测试、后续无标签阶段尚有额外工作量。
2. 资源探针发现GPU0/3/5各叠加了STAR实验进程468652/468631/468640，目录属于STAR-test/补实验/smooth_star_e_rx1_s3_e100_20260920_r01。对应六组最新完整轮为1041/1017/1037秒，而GPU1/2/4为276/268/273秒。8秒五次dmon读回：GPU0/3/5约84%—100%利用率；GPU1约14%—21%、GPU2约6%—38%、GPU4约17%—35%；GPU6/7空闲。该短窗不是全程利用率统计。
3. 当前同卡争用是有直接进程证据的主要可操作瓶颈。早期E15—22多个组共同出现尖峰，但没有那些历史时刻的GPU/CPU占用记录，不能把全部尖峰都归因于当前三个STAR PID。当前96逻辑CPU、load约9.7，也不能诊断为全机CPU容量耗尽。
4. 最新非同卡三组的训练batch耗时254—262秒，占epoch约94.8%；基础验证约5.1—5.3秒、重验证约8.6—8.9秒、其他约0.3秒。继续省日志/落盘已经不是主要方向。旧residual相近epoch重验证约79—81秒，新组约9秒，验证路径的优化已明显减少重复成本；不能将这9倍局部耗时变化全部单独归因于某一个开关。

## 加速是否真正生效

- resolved配置确认fast执行、buffer输入/输出、workers=2、梯度快照、增量日志和source_validation_reuse开启；进程探针确认每个训练主进程有两个信道worker。新动态namespace持续按epoch推进，并非固定训练IQ缓存。
- DAOT教师当前是identity_sequential：已经跳过教师非身份分支，但各教师视图仍分别前向；不是identity_batched。
- a1_runtime_fast=false，但a1_gradient_snapshot=true；不能因此称所有A1加速都关闭。未启用runtime_fast中的RC4无标签教师identity-only和部分批量读回路径。
- 固定验证/测试视图磁盘缓存路径已经配置，但六组共享evaluation_view_cache目录尚不存在，周期测试耗时均为0；E100前尚未触发该目标评估。source clean特征复用是另一项内存优化，不等于星地IQ磁盘缓存已经命中。将来的E100首次构建仍有成本，后续E110等可复用；模型预测仍须重新计算。
- 当前非有限梯度累计每组6—15步，约占已跑步数0.1%—0.25%，无非有限loss跳步；最近一次跳步在E21—E26之间，并非只有E1首步。最新完整轮全部更新成功，无持续全步跳过。保留异常包；不能因追求速度删除finite检查。

## 推荐下一步顺序

|优先级|措施|针对的浪费|边界与验证|
|---|---|---|---|
|1|先改善GPU调度与资源隔离|GPU0/3/5同卡任务使阶段收益被等待吞掉|后续任务优先GPU6/7；当前活跃任务不强迁移、不擅停。空闲显存不等于空闲算力；已有训练完整性优先|
|2|独立小窗口实测真实训练分段|目前只能知道95%在训练batch里，无法判定其中信道/同步/模型占比|已实现profiler但steps=0；下一次经授权在隔离诊断run分别测E21和后期激活阶段，预热后10—20步，固定输入/RNG，不改六组|
|3|先合并DAOT eval教师多视图前向，再审计RC4无标签教师identity-only|identity_sequential仍重复启动教师前向；a1_runtime_fast=false保留完整教师分支|现有identity_batched代码有eval与BatchNorm约束，仍须验证logits、loss、梯度及RNG等价；不要合并训练态student导致BN语义变化；预计收益尚未测|
|4|批量读回诊断标量，减少逐参数/逐记录.item同步|GPU1/2/4短窗利用率偏低，主Python约占单核，数据准备/同步可能限制流水线|按实际激活分支profile定位，保留路由、控制反馈、finite检查；梯度快照与DAOT scale批量读回已经开启，避免重复做同一优化|
|5|residual专用初始化与信道IPC优化|每record创建多个命名RNG；parallel_batch每次高级索引复制、pickle进程传输、结果拼接并等待|现有receiver仅在单batch按session复用，可审计扩展到worker内跨batch有界缓存；延迟构造无均衡路径不用的RNG；保留stable_seed派生、每流顺序和record状态。2→4worker或共享内存只做单项bench，不默认越多越快|
|6|E100后评估脱离训练进程、缓存构建单任务化|当前周期预测及CPU scorer同步阻塞训练；六组同时冷缓存可能重复生成同一视图|保留E100:10:200与clean+六环境全部覆盖，冻结checkpoint后排队评估；完成状态等待所有评分收齐。缓存原子写防损坏，但未防并发重复计算；可单独预生成或single-flight锁，不缓存训练views|

后两项不能提前报加速倍数。跨训练batch预取目前未实现，直接提前普通增强会改变全局Torch RNG与模型随机运算交错顺序；需要独立随机流方案及等价性验证，不能为速度固定整轮增强。全量预制数据也仍需保持按epoch/物理ID重新生成的训练多样性。

现在不建议减少六环境测试、降低困难样本比例、调整200epoch或改变损失来换取速度，这些会改变本次比较的科学问题。当前最优顺序是先资源隔离，再真实分段profile，随后优化教师前向与同步；进一步缓存/信道IPC跟随瓶颈证据。

## 可复核证据

- [逐epoch耗时CSV](evidence/speed_epoch_comparison_20260920.csv)：含六新组和旧四组全部完整epoch。
- [统计与覆盖清单](evidence/speed_analysis_20260920.json)：固定窗口、中位数、累计值、CSV/JSONL一致性及错误扫描。
- [当前资源探针](evidence/resource_probe_20260920.json)：PID/CWD/GPU映射与线程配置、CPU负载、缓存目录状态。
- 本地完整快照：evidence/speed_full_snapshot_20260920.json，73.5MB，包含完整结构化指标与stdout扫描结果，保留本地不进入Git；原日志/checkpoint均留服务器原路径。
- 实现定位：experiments/adv3b02_xuc/code/SSDG/train_ssdg.py中的_forward_daot_teacher_views、RC4 combined_w分支、source validation计时；cvsrffi/practical_parallel.py、practical_view_cache.py、a1_periodic_target.py与leo_practical/channel.py。

本轮仅分析和报告交付，没有修改服务器训练配置、进程或科学参数，没有启动新实验。
