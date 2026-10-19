# As 分类温度解耦：同初始化完整 E30

状态：REGISTERED 技术恢复。唯一 owner codex/root/temperature84；run 20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02。沿当前用户给出的附件及两个会话授权，优先回答同观测专家是否存在、弱证据分类动力学是否受尺度30限制。未发布的 Uroute83/noiseKL82 草稿保留，本轮不启动。

两条模型路径均从 seed2026092701 相同随机初始化开始，完整 scratchE20（13L+7U）再自身student/EMA独立追加E10，累计E30/6660更新；重置原AdamW/cos10，不从旧As20开始。每步原6次L128前向保持，.75ordinary+.25raw+.25As 的CE系数保持；CROSS_RX仅ordinary，原hiddenU/自身EMA/原门控保持，不新增epoch-bank/HQ/教师/额外U/对齐。每行5114880次L前向、426240次真实As观测；两行新增60模型轮/13320更新。U、验证、真实CP smoke、45k源诊断及正式评价成本分别记录。

T1/T6 都仅取消64个As槽的RDrop KL，ordinary和raw KL保持。T1为原尺度As CE；T6固定6×CE(logits/6)，保留原label_smoothing=.01。网络输出始终原6×160余弦头、scale30，所有参数按原流程联合训练，部署无温度操作。T6-T1才是温度增量，旧As/NoAs只作整配方参考。不同温度损失不可直接横比，日志另留原尺度NLL及解析logit梯度范数。

源数据不变：ManySig equalized1/256点/unitRMS，Fs25MHz，L6300/U56700/V27000，split392005，RX1/3/4/6/8，day1/2/3；无Ldev/heldRX。信道沿原fullpractical residual/post_sync/EQ关闭G1/refSNR30/reflect。训练fc2.462GHz，固定正式fc2.45GHz历史差异披露。使用合法实际角色句柄，不重复数据builder。

在T1训练前完成一次有界冻结诊断：PAMF原As30 seed2026092701、CSIL/MoPC原生默认头seed392005 FINAL200，读取同一个V1800面板、相同2draw的mid/lowurban IQ；统计教师对学生错、反向错误、共同错误、BLOCKED/锁定/新增噪声比分层。历史模型metadata已经确认scratch、固定final、完整role_ids精确相同；加载实际cp前再核对其携带的来源和role_ids。不同结构seed/训练预算不同，只声明同观测互补，不能作为严格匹配算法胜负。eval且无梯度，前后核对参数与BatchNorm状态。

另在原L内按TX×RX×day固定各20条（L1800），与既有V1800按同先验、独立seed20261019844生成2draw的mid/low观测，分别报告fresh L/V。只读已有cleanIQ，不读U真值、无更新；这比较已见物理样本的新信道迁移，不能代替实际已消费训练信道上的拟合率。全部新源诊断45000NN。诊断不自动授权蒸馏；只有同观测互补证据后另行固定最小配方。

两组均固定FINAL30并逐row自动clean+六种practical全部预测，所有prediction固定后独立scorer连接truth，全部指标/宏F1/最差RX/配对错误与实际成本保留。源条件预定：T6相对T1 midlow均值≥+.5pp，fullVclean≥−.1pp，只影响未来扩展。失败仍完整评分，不扫描温度/权重/种子补救；正式分数不回流选择。已暴露测试基准不声明新独立盲测。

环境：N607普通账户CVS-RFFI/fullFP32/CPU2。CWD PROJECT/releases/cvs_FcrossRX_AsTemperature_E30_20261019_r02；命令CVS-RFFI/python -u experiments/cvs_as_temperature84/runner.py。新不可变输出PROJECT/runs/20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02。资源队列按各owner总进程≤3/GPU且free≥12000MB，健康任务不改动。停止只限本run协议/角色/来源/非有限/碰撞/无法合法预测等系统技术失败，保留产物、先核对poststate，不自动重启。

聚焦验证：5项损失与隐藏标签负例通过，17文件AST通过。普通账户N607预检、输出未占用及资源核对VERIFIED；独立P0/P1审查进行中。launcher首步一次真实As30 checkpoint无query smoke，随后T1/T6完整6前向及两个segment路径合成检查，PASS直接训练。代码以Git commit、release_commit.txt固定；只校验一次发布归档，不增加报告或成员hash门槛。

独立P0/P1审查完成：未发现阻塞缺陷；只读核对损失/父EMA/预算/源比较/七视图评分，无额外审查轮次。

发布与真实检查已VERIFIED。代码commit facb98b73a524094c168243c4805c4760635e39d；归档比较、远端compile、真实CP及两segment完整路径合成检查3074NN PASS，直接进入队列；独立PID/CWD/argv/log核实。尚无新性能结论。


r01 T1在历史factory导入缺少paper_reproduction，已本地同指纹复现FAIL、临时加入历史根后PASS；未发生T1训练更新。r01 T6健康，绝不停止/重启/热改。r02仅从零启动原T1，T6仍读取原r01不可变输出，其config与运行中actualconfig已独立核对一致。新run新增训练30模型轮，矩阵累计仍两模型E30；failedr01诊断部分9000NN与3074smoke另计，保留，不作性能数据。历史路径只在模块导入期间加入sys.path并finally恢复。其他训练、seed、预算、源条件、固定正式IQ不变。

本缺陷唯一一次定点复审完成：导入修复与T1r02/T6r01复用路径未发现P0/P1；不扩大为重复全量审查。


源域同观测诊断已完成（45000NN/0更新）。V1800×2draw lowurban：PAMF59.0833%、CSIL77.4444%、MoPC78.3056%；CSIL对PAMF净修复661/3600，其中教师对而PAMF错835、反向174。BLOCKED低urban为32.7943%/61.4452%/62.9611%。低新增噪声比<3dB为27.3292%/53.7711%/56.5217%。不同模型seed及E30/E200预算差异已披露，这证明同观测下存在可供研究的专家优势，不证明蒸馏有效。

PAMF新信道的L/V lowurban为59.5278%/58.7222%，mid为74.0000%/73.1667%；相差分别.8056/.8333pp。这里是见过的物理样本在新信道下的表现，未测已消费的训练信道，不能笼统宣称优化欠拟合或物理上限。源报告见source_comparison.json。T1/T6仍按固定预算训练，未读取其正式测试指标。

日志字段解释：temperature_steps中的As_native_NLL实际为原尺度label_smoothing=.01的CE；不是逐步hard-label NLL。As_mean_logit_gradient_norm实际记录30×(两次平均softmax−平滑真值)的逐样本余弦梯度代理，不含batch均值/外层.25权重或其他目标的参数梯度。两种温度损失不可直接横比，分析按此语义解释，不更改健康训练。

温度两行E30/6660更新及全部七环境评分已完成。源门槛FAIL：T6−T1 midlow+1.5000pp，fullVclean−.3185185pp；固定T6关闭，不扩种子或改温度。实际两段完整结构化日志和stdout均已读取，原生CSV/JSON、L/U物理序列核对通过。正式指标继续等待KD源决定保存后统一读取。
