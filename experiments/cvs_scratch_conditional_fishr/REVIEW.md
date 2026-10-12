# 独立 P0/P1 发布审查

**PASS：未发现未闭合的 P0/P1。** 日期：2026-10-12。运行：`20261012-phase1-scratch-conditional-fishr-manysig-m9-r01`。

本次独立审查仅写本文件，未修改实现或登记、未提交、未访问远端或目标成绩、未使用 GPU，也未启动实验。依据本包 DESIGN 和已提交的 source-conditional-geometry/SOURCE_INTERPRETATION，核对当前运行路径及必要依赖；对相同的 energy6 训练、教师、风格和控制代码复用此前结论，并核对本包差异。性能推测不作为发布门槛。

## 实质核验

1. **矩阵与初始化。** 三方案 × 三模型 seed 共 9 行，学生随机原生身份网络，20 epoch／4440 更新，九行合计 39960 更新。学生初始化不读取 checkpoint；每行方法来源仍完整记录另外两个固定原 E20 教师，学生 seed 不混入自身教师。原生来源验证、源角色及完整祖先检查沿合法路径执行。不能将学生 scratch 等同于整个方法无预训练。
2. **条件交互和梯度。** A/B 在同 day 中独立抽取 6 TX × 5 RX × 2 的合法 L；批内 PID 唯一，批间可重合。每单元均值按 TX、RX 两维双中心化，损失为两个中心化结果逐坐标乘积均值。TX/RX 加性主效应位于零空间，保留负交叉估计。组合方案在同两次前向上另求有限 L 修正混合方差的 A/B 交叉项。虚拟仿射探针连接真实浅中层；CE 对真实未约减 CE 直接求导，`create_graph=True` 保留置信度及混合二阶导数，固定 RMS 仅以 Python 标量缩放。
3. **标定与单次总限幅。** 先完成 444 次身份更新，再以同 12 对独立 A/B 分两遍固定 RMS 和分项系数，共 48 次前向／2880 个 L 读出；首次训练辅助步为 448。每层各项分别测量自己的梯度，单项系数求导 48 次、组合 72 次。`_pair` 将已标定分项取均值；外层两系数严格为 1，随后一次 ramp、层均值除以 2，再执行一个 detached 联合标量 cap。没有重复乘分项系数或叠加两个独立 cap。cap 的身份参考含实际加权 CE、KL、风格等主目标，排除仅用于审计的 KL 分解，参数分区互斥完整并涵盖频率参数；复数范数实现使用绝对值。该约束限当前梯度，不声称限制 AdamW 位移。
4. **权限与实际预算。** 源 L6300 用于身份、风格、标定和辅助；U56700 只使用 IQ、RX/day、ID 和独立教师概率，拒绝 TX 真值。V 仅在既定 epoch 无梯度健康验证，固定 final20，不选模。每行 999 次辅助、1998 次辅助前向、119880 次辅助 L；3996 次额外风格/clean replay 和 19980 次候选 eval 前向。无辅助可训练头、暖头更新、EMA、卫星增强或 U 风格。额外前向计数与实际选中增强的计数分开。
5. **状态继承与预测门。** `fixed_conditional_sensitivity_v1` 显式保存目标、组成、444 边界、两层尺度、外层 1 和全部分项系数，进入 final checkpoint、辅助状态及资源证据。restore 拒绝旧 energy schema、错误目标、缺项或重复覆盖。预测入口将完整状态与本行 calibration.json 比对，再严格加载学生；预测不加载 U 教师缓存或用教师对目标推理。
6. **完整日志与评分。** 源分析器重算每项标定系数、分项加权均值、运行尺度和组成、联合限幅及全训练计数，允许负估计。逐行 source_frozen 后自动 clean prediction，再 exec 到独立 scorer 连接 truth；全部 9 行完成才写最终矩阵完成状态。每行 168000 个 query，合计 1512000 个决策。动态评分、collector/final report 行数和三个固定 energy6 跨批比较一致，缺 seed 不补选；目标结果不控制训练。
7. **发布与隔离。** 登记 9 个 config 与实际生成值完全相等。只读调用 configure 核对归档 61 项路径均存在，包含本包；preflight、dispatcher 和 scorer 绑定本包。唯一 owner、独立 run/release、逐行目录和日志独占创建；共享锁统计其他 CUDA 进程及启动占位，每 GPU 最多 3 个。未知启动状态不重复提交，失败保留产物、停止待启动项，不干预健康任务。

## 验证证据

已读取登记 `automation_reports/CV-SincNet/20261012-phase1-scratch-conditional-fishr-manysig-m9-r01/evidence/`：

- `conditional_fishr_math_cpu.json`、`conditional_fishr_math_cuda.json`：PASS。覆盖加性零空间、独立有限池期望、真实 CE/margin 一阶与混合二阶数值导数、三方案标定生命周期、同 A/B 复用、组合均值及状态拒绝。三方案组合目标的混合导数绝对误差最大约 `1.25e-12`；底层原始敏感度混合导数检查最大误差约 `6.43e-9`。
- `conditional_fishr_integration_cuda.json`：PASS。三方案原生全目标有限反向，活动步各 11 次前向；联合 cap 作用于完整 67 个身份参数张量，身份辅助 CE 梯度为零。此为本地合成集成，不是完整训练结果。
- `conditional_fishr_local.json`：PASS。全部 9 个预算、禁止初始化权重读取、各臂严格原生保存/加载、校准状态及来源负例。
- `reference_checks.json`：PASS。三个固定跨批对照及缺项处理。另对本包全部 Python 文件只读语法解析通过。

收尾定点复核：主代理已更正父包遗留的 parent20+20、逐样本归一化和暖头200等非执行说明，并同步全部9行登记；重新读回 exact configs 一致。reference_report 仅忽略上述说明字段，仍严格核对数值设置，独立执行其纯合成 self_check 为 PASS。evaluate 主方案和增量比较现明确为 CE_MOMENT_INTERACTION 与 CE_INTERACTION。更新后的 CPU 证据新增三臂真实 source.train 初始化至首目标边界检查，0身份更新且 PASS；dependency_checks 也为 PASS。未重复未改变的完整审查或训练检查。实际远端来源预检、发布和状态读回仍由唯一发布 owner 完成；本记录不宣称实验已发布或目标已达到。
