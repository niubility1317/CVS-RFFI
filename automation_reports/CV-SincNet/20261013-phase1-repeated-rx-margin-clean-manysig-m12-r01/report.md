# 可重复跨 RX 漂移上的鲁棒间隔：四组独立消融

按用户新报告（2026-10-13）停止新增解析不变量分支，本轮只改变训练期监督目标。CVS 原始输入、160 维身份表示、普通 cosine scale30 及 UNIFORM_SSL 设置保持；不叠加刚完成的跨 RX 对比。四组 × 三 seed 全部从头30轮，全部直接 clean truth-last，无源 V 晋级门槛。最终目标仍为三主 seed 均值84%，不是结果或调参反馈。

|组|方向|监督目标|
|---|---|---|
|MATCHED_BASE|无有效位移|原 CE＋SSL|
|RANDOM_ROBUST|实测位移的相同范数与 q，独立球面随机方向|逐竞争类别最小间隔|
|MEASURED_MEAN|三份独立样本重复估计的同 TX/day 跨 RX 位移|原始＋四候选平均 CE|
|MEASURED_ROBUST|与平均训练相同估计规则|逐竞争类别最小间隔|

L_s6300、U_s56700 TX真值隐藏、V27000不前向；源6TX×5RX（1/3/4/6/8）×day1/2/3。各统计单元按物理ID排序、序号mod3划分，当前训练样本只用另两份估计同TX同day的跨RX均值差。每轮开头用当前模型 eval/no_grad 刷新，恢复模块模式/RNG并检查buffer未变。所有四组、全部30轮均刷新，匹配这部分计算成本；前5轮仅原目标，a(e)=clip((e-5)/10,0,1)，第6轮0.1，第15轮1。

q=clip(2〈Δ2,Δ3〉/(||Δ2||²+||Δ3||²+1e-12),0,1)；平均位移先cap0.2再乘q、a。候选包含原始logits及normalize(u+δ)的四组普通cos logits。L_env=logsumexp(-min_candidate margin)，正确类margin为0；逐packet上界所有候选CE。总损失base+.25*(auxCE-CE)，平均组aux为五候选平均CE，鲁棒组aux为env；λ0/全零位移精确退化，不重复额外CE。统计张量detach，u与分类权重继续梯度。

每轮保存1080个q/范数/截断方向记录与三份90单元counts、总体分位数；每步记录有效位移和间隔威胁。每轮首/末步测净新增监督梯度对CE+SSL基线梯度的范数比和夹角，分别encoder/cosW（零新增梯度夹角null）。统计均值库172800bytes（168.75KiB）；不随checkpoint保留到query。每模型额外189000个L统计forward packet，12模型2268000；实际耗时记入报告。普通参数177025、6660update/model、79920总更新，无新增模型分支/波形候选前向。

先验参数、四组和全部三种子在本run target预测前冻结，不按target分数重排或重跑。C/D的统计定义相同，各自模型随训练改变，不共用陈旧坐标。独立三折减小自包含，但不等于留一RX验证；同TX/day不保证载荷/包位置可比。q不是置信概率，漂移不是纯RX因果。只验证有限经验候选，不保证任意未知RX。

每row末30轮冻结后自动clean168000预测，固定prediction后exec独立评分；总2016000决策/216指标。汇总全部seed/RX/day/TX准确率和Macro-F1，另报告相对MATCHED_BASE救回/新增错误及净改善，完整负结果保留。普通cos可在Phase2重拟合，但本轮clean不宣称Phase2部署。

唯一owner/root，普通N607账号；不可覆盖输出root、每GPU最多3总进程/至少12000MB空闲，保护其他owner。只按协议/数据/非法输出/非有限/无法完成合法预测等系统性技术失败停止，不以性能停止。单代码config提交/一次P0P1审查/聚焦负例及真实checkpoint无query smoke/单archiveSHA与remotecompile/启动PID-CWD-argv-GPU-log增长。精确命令、CWD、路径、GPU分配规则和expected artifacts见experiment.json。

|追溯ID|要求|实现|验证状态|边界|
|---|---|---|---|---|
|R1|同TX/day三折排除当前fold、独立重复性|drift.py|CPU/CUDA聚焦PASS（CUDA精确梯度检查使用deterministic卷积；训练数值策略不变）|观测统计非物理分解|
|R2|bounded四位移＋普通cos有限最坏间隔|objective.py|CPU/CUDA零强度/上界/梯度PASS|经验候选范围|
|R3|四组匹配UNIFORM_SSL/完整诊断/自动truthlast|design/source/score.py|实际UNIFORM_SSL loss/grad bitexact；12组synthetic aggregate PASS；唯一P0/P1及字符串修复聚焦复核PASS；真实运行待返回|不按target调参|

实际输入只读核实 VERIFIED：当前合法缓存 L day整数[1,2,3]、RX整数[1,3,4,6,8]、6300个物理ID；不采用嵌套历史reference_config的日期编码。90个cell各70个样本，三fold各24/23/23。

|ID|报告节|验收项|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|N1|3.1/3.2|同TX/day独立三折排除anchor|drift.py partition/estimates|verified|CPU/CUDA改anchor-fold不影响该fold位移；实际90cell核实|全部L仍训练|
|N2|3.2/3.3|q重复性、cap.2、随机等范数/q|drift.py|verified|重复/反向估计与random匹配PASS|q非置信概率|
|N3|5|当前坐标每轮刷新eval/no_grad、恢复mode/RNG/buffers|drift.py refresh/source.py|implemented|真实模型CPU/CUDA状态检查PASS；全30轮证据待返回|四arm同刷新|
|N4|3.3/3.4|原始＋4位移仅cos分类、有限逐竞争最小margin|objective.py|verified|真实cos160；逐packetenv≥maxCE检查PASS|不近似Jacobian/Gaussian|
|N5|3.4/5|base+.25(aux-CE)、零强度精确退化|objective.py|verified|CPU/CUDA实际loss/梯度bitexact，包括原SSL|不额外加完整CE|
|N6|5/6|四组×三primary从头30＋固定UNIFORM_SSL|design/source/registration|implemented|12configs一致；实际matchedSSL精确PASS；执行待返回|不叠加跨RX对比|
|N7|6|q分布及TX/RX/day、有效范数/间隔威胁|drift/source/objective|implemented|1080方向及step记录入口接通；真实日志待返回|内容/包位置未配对已披露|
|N8|6|净新增encoder/head梯度ratio与夹角|objective/source|implemented|真实模型两类梯度非零；每轮首/末60记录待返回|测netadded，零角null|
|N9|6|固定预测后独立truthlast与净救回/新增错分组|dispatch/predict/score|implemented|完整12row synthetic216指标/162净重分类PASS；真实产物待返回|RX/day字符串保留|
|N10|5|真实训练成本及query丢弃统计库|source/predict|implemented|query仅FINAL严格modelstate；30×6300cost计数接通|耗时待实测|
|N11|7|假设的可否定结果和有限观测边界|report/score|verified|六个预固定配对比较；不以sourceV或target调参|84未达，不保证未知RX|

CPU、CUDA、compileall与完整synthetic聚合检查已PASS。独立审查唯一P1为score把RX/day字符串转int，修复后定点复核PASS；无其他P0/P1。N607普通账号预flight于2026-10-13 10:57:45 CST确认8GPU空闲。代码/config由本次提交识别，发布包单SHA、远端compile与真实checkpoint no-query首步将在publisher自动执行；PASS后直接训练，无额外approval。
