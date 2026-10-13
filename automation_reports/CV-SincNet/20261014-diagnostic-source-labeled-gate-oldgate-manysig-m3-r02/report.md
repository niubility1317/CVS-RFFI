# 原门控的有标签源代理诊断

状态：本地协议负例与一次独立 P0/P1 审查均 PASS，待发布。完整矩阵以 experiment.json 为准。

研究问题：现有实际 U 接受率约 1.6%，不能把低覆盖或弱强一致解释为标签精度。使用已有 OLD_GATE_COS50 三个预定义 seed 的固定第 50 轮学生与各自 EMA，在合法 L_s=6300 上模拟原标签盲门控，分别测置信度、邻近、强弱一致、交集和物理邻居可用上限。标签只进入下游计数，模型与门控不接收 TX；U 真值不读，不为此放松旧门控。

每 seed 固定一次完整 L_s 遍历，batch256、drop_last=false，沿用 native u_unlabeled_shuffle，sampler/noise seed=modelseed+79021。同一 weak EMA 和原 native strong_noise 输入分别进入冻结学生的训练 Dropout/no_grad 与 eval/no_grad 路径。训练模式只模拟随机前向，不更新参数、buffers、EMA 或分类头。恒定同类高置信预测的 batch_neighbor 输出是相同盲元数据下的结构上限，不是新的训练规则。

数据沿用原 source contract、角色物理 ID、RX1/3/4/6/8×day0/1/2、6TX、L6300/U56700(hidden)/V27000，训练池标签比例10%。builder 仅核对既有 source handles，named target loaders 必须为空。诊断只前向 L，不前向 V/U，不读取 target IQ、预测或 truth。历史 student/ownEMA 来源按原 fixedfinal50/11100 和 EXACT_MATCH 核验；历史模型只读，不作新初始化或续训。

每 seed 输出 overall/RX/day/TX/RXday 的各级覆盖、正确数、精度和真/预测类别分布；零接受的精度为 null。共18900个独立 L 观察、56700次包前向、0 backbone/head/EMA更新。历史三模型150 epoch/33300更新只列为已有成本。L 是已见有标签训练样本，代理可能乐观；其覆盖与精度不能代替真实隐藏 U、未见 RX 或迁移指标，也不验收84%。

本地检查直接执行原置信度/邻近/一致门控，包含 TX 元数据拒绝、无邻居、强学生不同类、零接受精度、禁止 headfit 配置。一次独立审查仅限影响下次实际运行的 P0/P1。远端 smoke 严格加载同契约真实 checkpoint，零人工 IQ 前向，不查询数据；PASS 后立即顺序诊断。实际函数和当前分类头保持原版，没有引入新的读出器。

代码/config 以实际发布 Git commit 和 release_commit.txt 为唯一身份。环境为普通 N607 的既有 CVS-RFFI Python/full FP32，OMP/MKL/OpenBLAS=2；CWD 为 /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_label_gate_20261014_r02。入口：python -u -m experiments.cvs_source_label_gate.dispatch。输出根为 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-diagnostic-source-labeled-gate-oldgate-manysig-m3-r02，每行独占 GATE_PROXY-sSEED/summary.json 和 logs/GATE_PROXY-sSEED.log。owner 为 codex/root/source-label-gate-20261014。

按全局已有锁检查所有 owner，每 GPU 总进程少于3且空闲至少12000 MiB才在锁内建立本诊断 GPU 上下文；三行顺序执行，实际 GPU/PID/CWD/argv/log 写入产物。一次归档本地/远端 SHA、一次 compile 和上述 smoke，随后独立读回。已存在根不覆盖、不重发。错误角色/来源/EMA、计数、非有限值、参数buffer变化或执行失败停止后续新诊断，保留全部产物；不停止、重启或热改他人的健康 Sinc 训练。指标只用于解释，不改训练、选模、阈值或重跑。

独立审查结论：未发现直接 P0/P1。实际模型无 BatchNorm，训练模式无梯度前向不更新参数/buffers；历史自有 EMA、L 标签下游计量、全局容量锁、失败保留及入口均已核对。审查未修改文件或操作健康训练。

本r02是r01 fresh-worker模块路径技术修复，仅增加既有native_modules引导后再导入role helper，矩阵/采样seed/源模型/预算完全保持。r01源码与失败输出保留；修复前独立Python进程复现ModuleNotFoundError，修复后fresh-process gate checks PASS。该问题一次针对性复核 PASS；reviewer也用不经过wrapper的独立进程确认导入成功，没有改变模型/门控/权限/矩阵。
