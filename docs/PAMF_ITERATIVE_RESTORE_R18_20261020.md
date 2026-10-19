# R18：迭代锚点与观测求解位置的同容量比较

完整目标保持：独立恢复网络输出一份 IQ，原 PAMF 的 FFT/PA/统计/时域分支由该 IQ 一致计算；合法 L/U 全量、恢复训练不读 TX/F、原 F50/默认头冻结。正式 low urban≥70%，其余五个受损场景分别距同 F/seed 原始 raw clean≤5 pp；恢复 clean 损伤单列。此轮不是达标声明。

设计依据为用户六类报告 R3 的学习先验与软观测约束，以及 **R17 源域**证据：共享/不共享的 low urban为86.3889%/86.8333%，但只有不共享通过全部原源条件；前者在TX 14-7和mid urban条件失败，后者各TX均优于R16与R15。R17每步仍以coarse为先验中心，早期步的初始梯度很小；这些源机制证据提出“状态如何传递、CG放在哪里”的问题。R18不根据已公开目标分数选择结构、损失、阈值、排序或训练预算，也不称新的独立盲测。

两新行均保留R17不共享的三套Core9（2507526个可训练参数）、相同输入received/coarse/current/归一化真伴随创新/lognativevariance、相同冻结R07 coarse/action及variance/SCALE²。只继承R07源域训练上下文；不加载R17/R16/R15训练后prior作初始化。加载父后重置model2701；三Core从同一Core克隆，完整初始权重须与R17不共享epoch0逐值相同，仅作比较不用于初始化。**改变数值路径会改变初始完整输出，不要求新旧输出相同。**

|行|直接机制对照|神经前向 / CG|先验中心|
|---|---|---|---|
|current_anchor|每步在当前估计上累加学习修正|3次Core；各2次warmCG|current+SCALE×Core|
|terminal_dc|相对current_anchor只将观测求解移到末端|3次Core；末端6次warmCG|current+SCALE×Core|

current_anchor每步先验后执行2次软加权MAP，以前一步估计初始化；terminal_dc前两步令current=prior，第三步对最终prior执行6CG，以第二步估计初始化。两者都采用current残差通路；R17→current_anchor隔离先验锚点，current_anchor→terminal_dc隔离CG位置。两者均保留到早期Core的端到端梯度，不硬设Ax=r，不剪裁输出，不读取真实信道/噪声、场景或TX作为前向条件。三Core参数量及调用次数相同；3×2CG与1×6CG的重启/算子准备次数不同，**不宣称相同FLOPs或相同数值初始解**。本轮为报告R3的接收链机制适配，不是某篇原算法的逐式复现。

发布前的本地检查曾发现：若末端对照每步仍以coarse为中心，原Core零输出初始化使早期Core初始梯度为零。因此在实验登记/发布前将其改为与current_anchor相同的current残差通路，保留原权重初始化，不放宽各Core初始非零梯度检查。该本地设计修正没有训练、目标访问或正式结果。

各行重新scratch训练新prior20轮/4940更新，6300L/56700U全部使用，每行L126000/U1134000曝光；复用R07原20轮15%clean/85%residual缓存、model2701/data2001、batch256、AdamW2e-4→2e-5、warm247、WD1e-4、clip1、FP32，最终IQ NMSE唯一目标。新增各prior E20，组件coarse20+action20+prior20=E60；F50历史E50及R17历史参考训练成本另列。无新校准、F权重/输出损失、TX标签、教师或额外FFT损失。

固定最后20轮，source1800V各行七视图预测全部固定后独立CPU评分。每候选在原R16/native、R15/native条件之外，还必须通过R17不共享/native条件；四个固定源参考同时保护。原护栏保持：六场景平均NMSE≤各参考1.05倍、cleanNMSE≤.02、low/midurban分别≥各参考+.5 pp、clean相对源raw下降≤.1 pp、最差低urbanTX≥各参考−1 pp、其他五场景最差变化≥各参考−.5 pp。源V标签用于验收及固定排序，0V持久更新；整条选择不称完全无标签。

只在合格者中沿用原排序：低/中urban平均准确率最高、低urban最差TX最高、最差RX最高、最后名称字典序。无合格者0target；有合格者只冻结一个胜出网络，自动对其执行原168000物理记录×七视图、原F50/默认头推理和独立truth-last评分。R17正式预测作为**已有冻结对照**复用，不在新run重复预测；R18训练、源选择和推理不读取R17目标评分，只有新预测/评分完成后做同row对照分析。原rawclean84.6761904762%，其他五场景下限79.6761904762%，lowurban下限70%。

R17来源须核对方法commit、原source决策、unroll_unshared训练完成（20/4940、0TX/F/target训练、scratch）和source摘要；曾在完全冻结后正式评估不等于target训练污染，不能因其已评估而读取目标反馈选模。旧R16/R15保持原source-only参考核验；未改变的数据契约复用，不重复builder/哈希链。

N607池0/1/5/7择有容量卡，最多两条本run worker，每卡一条，全owner cap2/free≥12GB；共享capacity4锁到CUDA PID可见。独立不可覆盖root/release、唯一owner、普通账户。入口一次真实2Ls、新旧checkpoint/原F/各Core梯度/实际CG位置检查，PASS立即继续。技术失败保留产物，失败行/pending停止推进，健康任务继续，不自动重试；不以低成绩停机。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|A18-01|R3/R17源机制|current锚点三次2CG，当前状态可累积|models.py|verified|实际锚点与CG次数/梯度本地检查通过|同容量同输入|
|A18-02|容量/计算归因|三Core末端6CG对照|models.py|verified|中间无CG，三神经前向，参数与R17相同检查通过|非同FLOPs|
|A18-03|无监督与来源|全L/U、0TX/F/target、仅R07父，新prior scratch|runtime.py/training.py|verified|schema/来源负测及实际曝光|旧R17 epoch0只比较|
|A18-04|身份保真|四源参考护栏、固定排序、未选target阻断|evaluation.py|verified|R17参考非法target训练/读出角色/最差TX/排序负测通过|V标签选择单独披露|
|A18-05|可用checkpoint|新schema全状态、原F默认头、真实无query入口|runtime.py/controller.py|verified|新旧权重与epoch0匹配、真实样本输出及各Core梯度|不拟合分类头|
|A18-06|安全发布|Git/一次P0P1/不可覆盖/容量锁/原进程核实|controller.py/runner.py|verified|本地验证和N607实际poststate|不干扰其他owner|
|A18-07|完整证据|source全7、条件正式全7、TX/RX/day/损伤/成本|run报告|verified|完整日志、先预测后truth与已有R17对照复用|源成绩不代替正式目标|
|A18-08|真实性边界|新oracle反事实/多seed/独立物理族/逐细节真伪|后续确认|deferred|已有R14/R17边界，当前不宣称|不增加本轮发布门槛|

发布前验证：`python -X utf8 -m pytest tests/test_pamf_unrolled_restore.py tests/test_pamf_iterative_restore.py -q` 共 33 项通过。一次独立 P0/P1 审查发现目标评分收尾错误地把参数对象交给验收函数；新增回归先复现 TypeError，修复为读取已完成的目标摘要并保存 goal_assessment，定向复核确认解决，无其余 P0/P1。N607 普通账户预检通过，复用既有合法缓存及源参考，未打开 IQ/TX/F/target 数据。实际训练、真实 checkpoint 入口及发布后状态尚待执行。

实际状态：7 verified、1 deferred、0 rejected/blocked。SOURCE_NOT_QUALIFIED，选定 None。本轮没有新正式目标集成绩，持续目标尚未证实。训练全量曝光、0TX/F/target、实际checkpoint入口、源七视图、全部条件工作和运行退出已核实。最高剩余风险为单seed及逐细节身份真实性；[完整报告](../automation_reports/CV-SincNet/20261020-phase1-restoration-iterative-controls-selfsup-s2026092701-m2-r18/report.md)。
