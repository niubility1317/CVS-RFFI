# PAMF 无标签特征统计混合：固定源域假设

状态：VERIFIED_QUEUED。61项聚焦检查及一次独立P0/P1审查PASS；已发布并核实CPU队列控制器。等待同seed前序GPU工作完成，尚未实际source smoke、训练或目标预测。原先固定的科学配方和健康旧任务保持。

源依据：固定720条V中，既有RXPSM E30自身EMA的伪标签没有显示明显劣于监督E30；此前九模型加权类别分布及L-only原型覆盖正常。下一条研究因此保持原RXPSM E30状态，从低层接收风格变化检验U的作用，不换父来源、不强化类别配额、不加载G训练视图。

固定配方见hypothesis.json。在现有RXPSM U student forward的time_down和t2.pool输出上，各以0.5概率启用统计混合；alpha=0.1、eps=1e-6。以时间维均值、population标准差定义每条样本的风格，均值和标准差停止梯度。每批每条U仅可从现有软目标argmax相同、原始源RX不同、双方既有连续置信权重>=0.5的记录中均匀抽一个partner；无合法partner保持原样。每条记录采样Beta(.1,.1)系数，配对和系数在两个层位共享，层位启用独立。内容为自己的标准化特征，恢复尺度/偏移为自己与partner统计的凸组合；梯度只更新学生，不传到教师/统计。记录实际可配对及各层启用数，不能把配置概率当实际曝光。

现有U输入（半普通噪声/半Practical）、软CE、L-only跨RX原型/最终特征损失、ownEMA0.996、Gaussian0.999、阈下排除0.7、锐化0.5与原默认头联合训练全部保持；不加UCRIC损失、PAIR双批输入、辅助头、G训练输出或推理模块。每seed从同一原RXPSM E30学生/ownEMA/统计/原型独立追加20/4440，累计50/11100、13L+37LU。预期teacher/student U2097900、physical1048950，新风格调用4440批/1134000条；新增配对和统计运算实际成本披露。

GPU5/6/7固定seed2026092701/02/03。先等待同seed PAIREDU的全部14项预测完成并释放GPU，确认UCRIC已派发，再使用空出的第二个本任务名额；最多2条本任务/card，计入外部后最多3条GPU计算进程，派发时free>=10000MiB。CPU scorer等UCRIC终态后串行接续。实现和检查完成后才发布新不可覆盖root；真实E30无query smoke为第一步，PASS立即继续，唯一技术停止规则为相同技术失败两次停止新派发并保留健康任务。

最终学生自动完成raw及固定DURA v1.0各clean+六practical、168000物理样本/景，全部14项预测固定后独立truth-last。沿用同readout监督F20/普通PL F21三seed对照，完整报告逐seed/景/RX/TX/day、SD、配对差、DURA净纠错、曝光和实测成本。原raw双阈值同时满足才算goal达成，不以本假设、源V或组合参考代替。

参考：[MixStyle官方代码](https://github.com/KaiyangZhou/mixstyle-release)、[ICLR2021论文](https://openreview.net/forum?id=6xHJ37MVxxp)。借鉴低层统计混合和固定默认p/alpha；标签保护与1D层位是本项目待检验改动，不称完整复现或已验证收益。方法/父权重/系数/种子/资源在新目标访问前固定，原基准历史暴露如实保留，不根据PAIR/UCRIC后续目标成绩改动本方案或选择性重跑。


实现严格核对hypothesis.json与RECIPE相同。61项检查覆盖同类跨RX置信配对、统计凸组合/常量/无partner、只传学生内容梯度、临时hooks作用域与异常清理、隐藏U真值拒绝、原EMA/曝光状态往返、无新增模型参数、GPU双owner先后条件、双读出和真实方法名称、其他候选回归。实际E30 source-only smoke会临时强制两层启用以覆盖两个真实梯度分支；probe丢弃，正式3seed重新加载原E30，以各层0.5独立启用概率训练。未把local模拟检查当作真实训练或目标收益。


实际不可变执行commit `68ed7530a3c3caf96f7a54e1cfe4fc3c2a00d2f7`，一次发布包本地/远端SHA比较与远端编译PASS；独立读回CPU控制器PID/CWD/完整命令及实时存活状态。三seed行均QUEUED，没有新GPU子进程，source smoke尚未开始；不会把local检查或排队当成训练完成。队列等待同seed PAIREDU全14项预测完成释放GPU，UCRIC已派发后才可占第二个任务名额；source smoke PASS后自动开始原E30+20及两套七景预测/独立评分。首次source smoke临时强制两个特征层位覆盖，正式三seed重新载原E30、各层独立0.5启用，假设参数不变。证据见queue_verification.json。
