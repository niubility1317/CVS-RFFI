# F＋跨 RX：注入接收机 session 覆盖的冻结源域反证

run_id：20261017-diagnostic-FcrossRX-RXsession-fitL-s2026092701-m1-r01。全部固定源预测和独立复算已完成，科学FAIL；图表和最终交付进行中。唯一逐行配置见 experiment.json；本次仅一个冻结 ALL 模型，不是新 E20 训练或正式目标测试。

代码审计确认，原 PairedChannel 每包按 epoch/step/physical ID 刷新传播与底层噪声，而 receiver_for_session 始终由 receiver_seed20261016102 与 rx/day/eq 定义。四个 fit RX×三日只有12个注入硬件参数实现；同 session 跨 TX 稳定。既有波形自身未知 TX/RX 效应始终保留，不反演未知 TX。这只是覆盖事实，尚不是低 urban 错误的原因。

唯一假设：已经按这些12个 session 训练的 ALL 默认头，在同样传播与底层噪声下，对原 session 有可观察的优势；如果优势不存在，就没有依据将“注入 session 覆盖不足”作为下一训练方向。它属于 As 的覆盖诊断，不是已验证 IQ 泛化增强 Ag。

## 固定对照与判据

仅用 P15 ALL 的 FINAL20（实际运行 f788309f5f9e27689ec1e98ae99a6bd83237e1f9）：F＋跨 RX 同 TX SupCon、三 SIG_TIME、统一 Sinc、NoExplicitCFO、joint160/defaultcos30。原 scratch E20／4440 联合更新／fit L5040 U45360 V21600、hidden U／ownEMA、FINAL-only、无祖先/目标参与的来源，在加载前与实际 config/init/completion/source_fold/freeze/result 核对。不是 E50/E200 权重，不替换或另训头。

冻结评估只消费父训练已见 fit L5040，RX3/4/6/8、day1/2/3、全部6TX、72格各70。原缓存反序列化 L6300 及未用 V/历史特征，RX1的1260包在选用IQ前排除；U只有opaque角色ID，没有U IQ/真值，V/held/target不参与。实际 source_cache ID 是 opaque 字符串，不能从中解码TX或物理字段；RX/day来自已验证L元数据，eq1沿用现有载体契约。只保存合法可见L的ID→TX/RX/day供独立读者核对，不新建权限链或解封隐藏真值。

|视图|源IQ|传播/底层噪声|注入RX参数|额外训练|
|---|---|---|---|---:|
|clean|原缓存RMS IQ|无新增传播|保留原录音效应|0|
|original|同一raw经原normalize_input|每包每场景一个固定命名实现|原rx/day/eq session|0|
|bank1/2/3|同上|与original完全相同H/底层噪声seed|同分布三个固定virtual bank后缀，跨TX稳定|0|
|original replay控制|同上|原seed重放|原session|无额外NN|

只比较 mid/low urban 两场景。实际入口沿用P15不可变后端的experiments/adv3b02_xuc/code/leo_practical，运行保存__file__/实际全六配置；fs25MHz、fc2.462GHz、IQ256、residual/post_sync、EQoff G=1、reference30、reflect、接收质量/相位跟踪/IQ补偿/AGC保持。四种session共用H/数值边界/质量/锁定状态；RX广义线性处理会相应改变处理后噪声和AGC，不能把共享底层噪声说成输出噪声逐点相同。新增噪声SNR不等于总SNR。不得提高SNR、改测试参数或删难包。

在真实IQ访问前固定，两个场景均必须满足：original准确率比三bank均值至少高1个百分点；original到三bank的平均hard判决变化至少3%；任何单bank比original高不得超过0.5个百分点。全部45360判决／10080物理记录／50400原生process／48稳定session、原重放0MSE、状态/预算/模型冻结检查也必须满足。源判断独立全量复算；不根据某个bank最差分数选新的训练session。

FAIL关闭这个固定session覆盖理由，不换bank、seed、场景、RX或阈值救跑。PASS也只支持另行预登记匹配训练，不能证明Ag、泛化收益或因果分离；没有自动训练。父已见fitL结果不冒充heldRX、V或target表现，三个物理bank不是三个模型seed。

## 执行与产物

0新model-epoch／优化器／教师／EMA／诊断头拟合；1个已有CP读取，launcher第一步strict load后2个合成包no-query smoke，PASS立即继续，合计45362 NN。5040 clean＋5040×两场景×四bank＝45360源判决；每包每场景4次原生process加1次replay，共50400调用。历史ALL训练成本另列E20／4440更新／5823318 NN，不作为本轮训练或继承收益。

普通N607既有CVS-RFFI环境，CPU2线程，CUDA_VISIBLE_DEVICES为空，单进程，不使用GPU训练容量、不装包或干预其他任务。CWD：PROJECT/releases/cvs_FcrossRX_RXsession_fitL_20261017_r01；命令：CVS-RFFI/bin/python -u PROJECT/releases/cvs_FcrossRX_RXsession_fitL_20261017_r01/experiments/cvs_rxsession_probe17/runner.py。精确可执行路径、CWD、run/log根在experiment.json；不可变run根PROJECT/runs/20261017-diagnostic-FcrossRX-RXsession-fitL-s2026092701-m1-r01，日志在release/dispatcher.log，本地证据E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017。唯一owner codex/root/FcrossRX-RXsession-source-20261017。

只因本run角色/来源、非有限、冻结状态、预算、输出碰撞或不可启动等系统故障停止并保留产物，不自动重试；低分不是技术故障。按项目最小流程完成一个Git代码/config提交、一次限定P0/P1、一次普通N607 preflight、一个release归档传输比较与remote compile、真实CP smoke、启动后PID/CWD/argv/CUDA/log增长或自然终态独立核验。

本地51个协议反例、6个科学反例、30次原生合成process PASS，0真实IQ/CP/NN/更新。一次审查发现并修复两个直接错误：四RX source fold实际标记与原carrier不同；fit source_cache采用opaque ID而非held readout的五字段元组。新增合法四RX来源通过、旧标记/污染/角色混用拒绝，以及opaque ID越界/重复缺失回归；未启动真实run补救。本候选只进行原问题定向复核，不重复全审。

产物：实际配置/模块/默认头、合法fitL标签与opaqueID元数据、四bank全部logits/argmax/margin30、完整物理元数据/48session及replay检查、全部40批stdout、独立准确率/F1/CE/配对修复损坏/固定clean正确集保持和TX/RX/day/状态/质量分层、输入变化/成本/图/同row报告登记。独立读者只读取冻结预测与保存元数据，不重新打开IQ或运行网络。IQ、权重、逐包ID/原预测日志、私有机器信息及生成图不提交Git。

REJECTED_EXTRA_GATE：通用skill的额外设计审批、固定spec文件、hash/signature或重复review不在项目八项白名单；当前已有授权内继续最小源反证。模型、分类头、L/U/V和正式clean＋六practical保持。当前无新的识别收益，Ag独立价值、As稳定性、B00/B10/B01/B11交互、正式low>60%、suburban接近clean与跨seed联合目标均未完成，保持ACTIVE。


实际发布：代码31e63c58cfea58cb94cf7259f9ab87941cf59348已推送并独立核验OID identical。一次普通N607预检、单archive传输比较、remotecompile通过；真实匹配CP2合成NN smoke PASS后立即进入fitL。原PID199424的CWD/argv/CUDA隐藏和日志276→297字节均独立读回，13/40批，14978NN／16640物理调用，0更新/target，无异常；该条为启动读回时状态；最终终态见下文。

完整原进程已自然终止，40/40批／45362NN／50400物理调用／48session，主体100.299359s；全部45360判决和10080物理记录独立复算一致。mid原session73.253968%、三bank73.273810/73.273810/73.313492%，平均hard变化0.449735%；low原57.738095%、三bank57.698413/57.777778/57.817460%，平均hard变化0.978836%。原session对三bank均值优势分别−0.033069/−0.026455pp，均未达原+1pp与3%变化条件，固定理由FAIL关闭，无后继训练。当前图表准备不修改运行release。
