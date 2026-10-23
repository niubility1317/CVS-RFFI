# PAMF 无标签特征统计混合：固定源域假设

状态：VERIFIED_RUNNING。61 项聚焦检查、一次独立 P0/P1 审查及真实 E30 source-only 梯度检查均 PASS；三个 seed 已按原配方实际续训，健康 UCRIC 并行保留，原目标仍 ACTIVE。

源依据：固定720条V中，既有RXPSM E30自身EMA的伪标签没有显示明显劣于监督E30；此前九模型加权类别分布及L-only原型覆盖正常。下一条研究因此保持原RXPSM E30状态，从低层接收风格变化检验U的作用，不换父来源、不强化类别配额、不加载G训练视图。

固定配方见hypothesis.json。在现有RXPSM U student forward的time_down和t2.pool输出上，各以0.5概率启用统计混合；alpha=0.1、eps=1e-6。以时间维均值、population标准差定义每条样本的风格，均值和标准差停止梯度。每批每条U仅可从现有软目标argmax相同、原始源RX不同、双方既有连续置信权重>=0.5的记录中均匀抽一个partner；无合法partner保持原样。每条记录采样Beta(.1,.1)系数，配对和系数在两个层位共享，层位启用独立。内容为自己的标准化特征，恢复尺度/偏移为自己与partner统计的凸组合；梯度只更新学生，不传到教师/统计。记录实际可配对及各层启用数，不能把配置概率当实际曝光。

现有U输入（半普通噪声/半Practical）、软CE、L-only跨RX原型/最终特征损失、ownEMA0.996、Gaussian0.999、阈下排除0.7、锐化0.5与原默认头联合训练全部保持；不加UCRIC损失、PAIR双批输入、辅助头、G训练输出或推理模块。每seed从同一原RXPSM E30学生/ownEMA/统计/原型独立追加20/4440，累计50/11100、13L+37LU。预期teacher/student U2097900、physical1048950，新风格调用4440批/1134000条；新增配对和统计运算实际成本披露。

GPU5/6/7固定seed2026092701/02/03。先等待同seed PAIREDU的全部14项预测完成并释放GPU，确认UCRIC已派发，再使用空出的第二个本任务名额；最多2条本任务/card，计入外部后最多3条GPU计算进程，派发时free>=10000MiB。CPU scorer等UCRIC终态后串行接续。实现和检查完成后才发布新不可覆盖root；真实E30无query smoke为第一步，PASS立即继续，唯一技术停止规则为相同技术失败两次停止新派发并保留健康任务。

最终学生自动完成raw及固定DURA v1.0各clean+六practical、168000物理样本/景，全部14项预测固定后独立truth-last。沿用同readout监督F20/普通PL F21三seed对照，完整报告逐seed/景/RX/TX/day、SD、配对差、DURA净纠错、曝光和实测成本。原raw双阈值同时满足才算goal达成，不以本假设、源V或组合参考代替。

参考：[MixStyle官方代码](https://github.com/KaiyangZhou/mixstyle-release)、[ICLR2021论文](https://openreview.net/forum?id=6xHJ37MVxxp)。借鉴低层统计混合和固定默认p/alpha；标签保护与1D层位是本项目待检验改动，不称完整复现或已验证收益。方法/父权重/系数/种子/资源在新目标访问前固定，原基准历史暴露如实保留，不根据PAIR/UCRIC后续目标成绩改动本方案或选择性重跑。


实现严格核对hypothesis.json与RECIPE相同。61项检查覆盖同类跨RX置信配对、统计凸组合/常量/无partner、只传学生内容梯度、临时hooks作用域与异常清理、隐藏U真值拒绝、原EMA/曝光状态往返、无新增模型参数、GPU双owner先后条件、双读出和真实方法名称、其他候选回归。实际E30 source-only smoke会临时强制两层启用以覆盖两个真实梯度分支；probe丢弃，正式3seed重新加载原E30，以各层0.5独立启用概率训练。未把local模拟检查当作真实训练或目标收益。


实际不可变执行commit `68ed7530a3c3caf96f7a54e1cfe4fc3c2a00d2f7`，一次发布包本地/远端SHA比较与远端编译PASS；独立读回CPU控制器PID/CWD/完整命令及实时存活状态。三seed行均QUEUED，没有新GPU子进程，source smoke尚未开始；不会把local检查或排队当成训练完成。队列等待同seed PAIREDU全14项预测完成释放GPU，UCRIC已派发后才可占第二个任务名额；source smoke PASS后自动开始原E30+20及两套七景预测/独立评分。首次source smoke临时强制两个特征层位覆盖，正式三seed重新载原E30、各层独立0.5启用，假设参数不变。证据见queue_verification.json。


## 已核实实际训练

执行 commit `68ed7530a3c3caf96f7a54e1cfe4fc3c2a00d2f7`；独立核实控制器和三条 worker 的 PID／CWD／完整命令、实际 GPU 与真实 step 日志增长。真实同契约 E30 学生／自身 EMA probe 通过 768 条 L、256 条 U 前向／反向，246 条可配对 U，浅、中层各应用 246 条，有限非零梯度，教师无梯度且临时 hooks 清理。smoke 为覆盖梯度暂时强制两个层位，probe 丢弃；正式三 seed 重新加载原 E30，仍各层独立 0.5 启用。逐条核对全部已刷盘的完整 step 前缀，可配对及两层实际启用均非零、曝光计数准确；此时 gzip 未关闭，只是前缀证据，不宣称 4440 步完成或收益。源运行证据见 source_activation.json。

PAIR 三 seed 全部 14 项预测完成后释放了原槽位；UMIX 在 GPU5／6／7 各与健康 UCRIC 共占本任务两个名额，含其他 owner 的卡上总计算进程最多三个。旧任务不热改或停止，后续固定 E50 学生自动完成 raw／同一 DURA 各七景并独立评分。


## 三seed源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：三行各新增20轮/4440步，累计E50/11100；逐条读取全部13320条新增闭合step记录，源U曝光2097900、物理U曝光1048950、学生U视图2097900和自身EMA11100次均符合预定，父前6660步复用原完整核验记录。无U真值/目标训练或另训头。完整源证据及已发生父阶段/增量/累计时间见source_training_complete.json；程序经过时间来自共享GPU，不能等同独占GPU小时。

模型已按固定最终学生冻结，并已自动进入raw与固定DURA各七景预测；这是源训练完成，正式预测/独立评分尚未全部齐，不宣称整体实验完成或目标达成。部分目标成绩仍不读取，三seed全矩阵完成后统一分析。


## 完整目标测试结果

ANALYZED：三个 seed 的 raw 与固定 DURA 各七景、共 42 项读出／场景均完成独立 truth-last 评分，六个匹配监督／普通 PL 对照模型完整复用。逐景／逐 seed／SD／Macro-F1／RX／TX／day／配对修复损伤、实际曝光与成本见 target_results.md、target_summary.json、target_per_seed.csv、target_strata.csv、target_paired_raw.csv、restoration_effects.json、evaluation_completion.json。原 raw 双门槛尚未满足，研发目标仍 ACTIVE。

raw clean：84.362896825% ± 0.397129695%（三个模型 seed 样本 SD）。
DURA-LU clean：84.433134921% ± 0.447926192%（三个模型 seed 样本 SD）。
raw six_mean：72.000892857% ± 1.012286362%（三个模型 seed 样本 SD）。
DURA-LU six_mean：77.554133598% ± 0.232519585%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。
