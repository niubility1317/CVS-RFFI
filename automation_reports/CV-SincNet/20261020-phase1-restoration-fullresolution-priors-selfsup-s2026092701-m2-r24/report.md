# 全分辨率 TCN 与注意力三步恢复 R24

状态：RUNNING，真实入口与两条 GPU worker 已核实。方法 commit `da5b9d52819ccd40f8b59dbfdefebff3bd3c2230` 已推送并核对远端 OID。37项聚焦检查、compile/help和一次独立P0/P1 PASS。真实checkpoint入口在远端首步执行，PASS立即继续。


当前状态：LOCAL_VERIFIED。两个新先验与一个复用源控制的完整训练链路已通过本地聚焦验证，尚未发布。以下“先前原型”保留13项检查的历史；当前实际矩阵与验收追踪以文末“完整实验设计”为准，不把原型检查当作训练或性能证据。

## 先前原型记录

状态：LOCAL_PROTOTYPE_VERIFIED，尚无源训练、远端发布或性能结论。R23 正常运行，不更改其代码、矩阵或资源。该备选依据 R17/R22 的源域结构与结果提出，不使用任何正式测试成绩设计。

用户要求比较多种网络。已比较的 U-Net、TCN、时频、扩散/流匹配及解析约束各有独立记录。本轮新增另一结构假设：逐样点全分辨率 token 的内容相关注意力，配合局部卷积，可能补充多尺度卷积的修正能力；这只是待证伪假设，不预先宣称保留了更多 TX 指纹。

备选矩阵为同一冻结 R07 上下文下的 `unet_control` 与 `attention_prior`。均为三个不共享先验、每步两次 warm CG、coarse 锚定、最后输出同一份 IQ；原 PAMF 再一致计算 FFT/PA/统计/时域量。只继承合规 R02/R07，不继承 R23 coarse 更新或任何后续训练 prior。两种结构先验头均零初始化，因此完整初始输出相同；隐藏参数形状不同，不能称完整初始权重相同。

注意力先验接收原九通道 `[received,coarse,current,normalized innovation,log variance]`。9→128 逐点投影，全长256，不做 patch、下采样或跨样本混合。固定0.1倍正弦位置量；6个 pre-norm block，每个含4头样点自注意力、5点深度卷积＋逐点混合、扩展率2的SiLU MLP，三条残差各缩放0.5；最终LayerNorm和零初始化128→2头。三个stage分别持有独立参数。训练使用非重入activation checkpoint减少保留的内部激活，增加重算；推理不重算。没有Dropout、BatchNorm或持久输入统计。位置量是固定buffer，不使用RX、场景、TX、F、clean或其他样本信息。

本地原型只验证数学与接口。后续真实候选须另行完成完整runtime/schema、训练/评分路由、一次独立P0/P1、登记/commit、真实checkpoint首步与不可覆盖发布，再可训练。当前不把这些未来工作当作R23阻断条件。预期仍为全6300L/56700U、TX无标签模拟配对IQNMSE、0训练好F/TX/target、原冻结F50/defaulthead源七视图资格和条件正式确认；最终预算与矩阵在任何新运行前固定。无准确率目标放宽，正式low urban≥70%，其他五场景各≥79.6761904762%。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T24-01|多网络设计|全分辨率注意力＋局部卷积，单IQ输出|models.py|verified|形状/零头/参数检查|报告思路扩展，并非原六类精确复现|
|T24-02|样本独立|注意力只在单包时间轴，无跨包或持久更新|models.py|verified|逐包/置换/状态检查|不使用全局query关系|
|T24-03|可训练性与成本|checkpoint前后输出/梯度一致，三个stage可达|models.py/tests|verified|实际反传/逐层梯度|增加重算成本，参数量实报|
|T24-04|无监督权限|仅接入原冻结R07、clean只训练目标|完整runtime/training|deferred|尚未接入真实训练|当前随机输入原型不等于数据契约验证|
|T24-05|实验验收|登记/真实checkpoint/源与条件正式评分|完整pipeline|deferred|尚无运行|不阻断R23|
|T24-06|身份真实性|逐TX/RX/day及独立物理族/多seed|后续合格实验|deferred|尚无性能证据|不能由接口测试证明|

当前3 verified、0 pending、3 deferred、0 rejected、0 blocked。13项聚焦检查及compile通过，仅证明本地原型接口/梯度/单包属性，不证明真实数据或识别效果。N607 Torch2.1和batch256内存/速度未验证，完整pipeline和一次候选审查留待正式登记前完成；不阻断R23。

实际随机初始化结构参数：{"unet_control": {"parameters": 4665073, "trainable": 2507526, "position_buffer_elements": 0}, "attention_prior": {"parameters": 4863217, "trainable": 2705670, "position_buffer_elements": 98304}}。注意力三个prior共2705670可训练参数，比UNet三个prior2507526增加约7.9%；全部模型参数分别4863217与4665073。新增位置buffer共98304个FP32值（393216字节），不作为可训练参数隐去，也不等于实测峰值显存。
验证命令：`python -X utf8 -m pytest tests/test_pamf_attention_restore.py -q --maxfail=1`、`python -X utf8 -m compileall -q experiments/pamf_attention_restore`。先观察缺少实现的失败，再实现并通过13项检查；未运行源数据训练或新正式评分。

## 完整实验设计

持续目标不变：原冻结F50 seed2026092701/defaulthead，正式low urban≥70%；其余五场景各≥79.6761904762%（原raw clean84.6761904762%−5pp），restored clean损伤另列。原PAMF从同一恢复IQ生成全部FFT/PA/统计/时域分支。目标成绩不反馈训练、结构、资格、排序或选择性重跑。

首轮R01已经测试四点token、64token/64宽/三层的单次残差Transformer，整轮未达稳定身份恢复，不能称R24首次Transformer。其固定训练观测、复合损失、F读出也不同。R15已测试单步四输入TCN＋FFT辅助项；R24为九输入三步、仅IQNMSE，跨轮不作单因素归因。[R01历史](../automation_reports/CV-SincNet/20261019-phase1-pamf-restoration-selfsup-manysig-m7-r01/report.md)。

实际新增训练行是`tcn_prior`、`attention_prior`。复用R23 `prior_only`作固定同数据/同预算U-Net源控制，不重复训练，不把其prior或R23 joint coarse用于新模型初始化。R23控制已经完整20/4940、全L6300/U56700、coarse/estimator更新0、训练TX/F/target读取0，原F50/defaulthead、固定1800×七视图全部预测后truth。只接受`20261020-phase1-restoration-joint-coarse-selfsup-s2026092701-m2-r23` / `d43dbab326d0b1e2cb6fb7faa23dc9a3ec25a025`的匹配源控制；这是先期固定控制，不称同期确定性重复。R23整体0target且所属进程已退出。

两行只继承原R07冻结上下文（R02 coarse20＋R07作用估计20），三个prior从头、不共享参数，每步2次warm CG、coarse锚定。新增TCN为9→128的5点stem、6个原Block(128,128)、dilation1/2/4/8/16/32、长度始终256、零128→2头。名义卷积感受野269，GroupNorm仍作用于本包全窗，不称严格局部或复数相位等变。注意力结构与上节原型相同。零头使完整初始输出相同，隐藏结构不同，不能称完整初始权重、参数量或FLOPs严格匹配。

两者均接收received2/coarse2/current2/归一化adjoint创新2/log原生方差1。估计作用和白输入方差代理的权重冻结，不把代理当作全噪声协方差或可辨识真信道。全原6300L/56700U、R07原二十轮缓存、15%clean/85%residual、model2026092701/data2026102001/cache2026101912；每行E20/4940、batch256、L126000/U1134000，AdamW2e-4→2e-5、warm247/cos、WD1e-4、clip1、FP32/noTF32、固定末轮。只有detached clean逐包IQNMSE，无FFT/解析/身份/PL/F引导或V拟合、target训练。名称为TX无标签模拟配对自监督，非无clean训练或整条识别链无监督。组件顺序coarse20+action20+prior20；原F50监督E50另列。

原五参考native/R15/R16/R17/R19保护不变，并相对复用R23控制使用同一套成对波形/身份保护。每行1800×七视图全部预测后独立CPU连接source truth；两行完成后按low/mid均值、最差TX、最差RX、名称固定排序，至多一行。无资格0target。合格者自动原168000×七视图、原F50/defaulthead、全部预测后独立truth-last及既定配对身份诊断。完整报告三郊区/三urban/clean、TX/RX/day、修复/损坏、波形/配对clean特征与成本。已暴露基准不称新盲测。

实际代码commit、一次独立P0P1、最小登记后，新不可覆盖release/run，N607普通账户，一次预检/单归档比对/编译。launcher首步真实2Ls/checkpoint/noquery smoke，PASS立即继续。入口核实R23退出；GPU池0/1/5/7、最多两条本任务worker、每卡一条、全owner cap2/free12GB，共享capacity锁至CUDA PID可见。非有限、错误数据/来源、输出冲突、非法或不完整prediction按预登记技术规则停止失败行、保留产物与健康其他任务，不因低分热改或重启。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|R24-01|多网络设计|全分辨率TCN/注意力，3步6CG单IQ|models.py|verified|形状/零头/初始输出/参数|报告扩展，非首次Transformer|
|R24-02|样本独立|单包、无跨包/持久统计|models.py|verified|逐包/置换/状态|GroupNorm作用于本包全窗|
|R24-03|梯度与成本|checkpoint梯度一致，三stage可训练|models.py/training.py|verified|实际反传/参数/成本|原型13项不覆盖新TCN|
|R24-04|无监督/来源|R02/R07初始化，R23仅控制，0TX/F/target训练|runtime.py/evaluation.py|implemented|来源/schema/角色负测/真实checkpoint|不重验未变数据|
|R24-05|源资格/隔离|六参考保护、固定排序及条件正式|evaluation.py/controller.py|implemented|负测/全部预测后truth|无资格0target|
|R24-06|资源/发布|登记/commit/smoke/review/队列/退出|runner/controller/登记|pending|实际GPU/PID/日志及退出|不干扰其他owner|
|R24-07|完整结果|源及条件正式、逐TX/RX/day、成本/Git|report/summary|pending|独立评分/本地完整权重|不以NMSE代替身份|
|R24-08|身份真实性|多seed/独立物理族/逐细节TX归属|后续合格实验|deferred|尚无性能证据|不作为早期启动门槛|

完整设计当前3 verified、2 implemented、2 pending、1 deferred、0 rejected、0 blocked。37项聚焦检查、compile/help及一次独立P0/P1 PASS，真实checkpoint首步与N607训练尚待执行。最高未证实项是真实输入对应TX细节及跨RX泛化。REJECTED_EXTRA_GATE：通用全仓测试、重复review、签名和额外数据重验不新增为门槛。

当前完整结构参数：{"unet_control": {"parameters": 4665073, "prior_parameters": 2507526, "position_buffer_elements": 0}, "tcn_prior": {"parameters": 4549105, "prior_parameters": 2391558, "position_buffer_elements": 0}, "attention_prior": {"parameters": 4863217, "prior_parameters": 2705670, "position_buffer_elements": 98304}}。TCN和attention均为3个不共享prior，参数/FLOPs不严格匹配，隐藏初始化不同。位置buffer单独列出，非实测峰值显存。
当前完整验证：`python -X utf8 -m pytest tests/test_pamf_attention_restore.py -q --maxfail=1`（37 passed）；`python -X utf8 -m compileall -q experiments/pamf_attention_restore`；`python -X utf8 -m experiments.pamf_attention_restore.runner --help`。只有随机输入/临时元数据的本地检查，未伪称真实恢复或识别证据。


完整矩阵、命令、环境/CWD、路径、GPU和停止规则见[登记](experiment.json)。


发布后状态：VERIFIED。单归档传输比对、远端编译和真实两条源样本 checkpoint/defaultF 入口通过。原R07上下文一致、完整初始输出相同，隐藏结构不同；三步 prior 梯度有限非零，全部上下文无参数梯度。R23控制完整权重仅严格加载检查，不进入新prior初始化。两条真实 GPU worker 的 PID/CWD/argv/GPU 和日志增长已核实，每卡一条，未干预其他 owner。全LU训练、源七视图及有条件正式确认自动执行；当前没有新的性能结论。


一行完整源评分已返回：[tcn_prior局部报告](tcn_prior_source_interim.md)。矩阵仍RUNNING，不改变其他行或最终选择规则。
