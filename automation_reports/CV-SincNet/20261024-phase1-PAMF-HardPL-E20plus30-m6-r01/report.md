# E20 后更早利用高纯度无标签源样本

状态：LOCAL_VERIFIED_NOT_LAUNCHED。48 项聚焦检查及一次独立 P0/P1 PASS；首个 GPU 步骤为真实原 F21 E20 两臂检查和新的 E20 源配对身份诊断。

原 F21 E20 完整源 V 各27000条，阈值0.95接受正确率99.9002%／99.8687%／99.9164%，覆盖92.7926%／93.0741%／93.0667%。仅冻结默认头CPU诊断，无状态更新／U真值／目标读取。原E20已有足够高的源伪标签纯度，故预先固定更早启用机制，延长其在同一E50预算中的作用时间；不以当前MS/NAMS目标成绩改变参数、父模型或种子。

两臂均使用同seed原F21固定scratch E20（13L＋7普通PL），独立追加原生10＋20轮／6660步，累计E50／11100、13L＋37LU；原E30状态不读取为初始化。第一臂HPL_E20_LS2只用普通硬PL，权重2／门控0.95／U标签平滑0.01／一次raw U学生前向。第二臂HPL_E20_NAMS2沿已固定NoAs＋MixStyle完整U配方：六前向、ordinary/raw RDrop1、As KL0、LS0.01、浅中层跨RX0.1、CE剂量除1.25；U As输入exact raw，跳过U信道生成；accepted-only同硬伪类不同源RX的统计混合，每批六前向共享配对／lambda／层开关，time_down／t2.pool独立p0.5、Beta0.1、eps1e-6，先混合后捕获对比特征。池化T1层为特征均值插值，不称非退化时间方差。原L六前向／信道及默认头联合训练保持，无EMA、原型、U真值、G训练、新推理参数或另训头。

保留原20／10／20三段原生优化器及增强日程，不创建13轮短cosine父模型。每模型新U1701000、6660调用，累计U／教师evalU2097900；LS学生U2097900，NAMS学生U10602900、rawAs850500、style调用39960／共享计划6660。新机制由20轮增加至30轮，模型总轮次不增加；学生前向与实际父／增量／累计经过时间单列，不宣称相同FLOP或把共享GPU墙钟视为独占小时。

运行前严格检查原E20数据契约／seed／run／row、scratch父源、实际checkpoint epoch20／4440、无nativeU／EMA／外部teacher与U真值／target；禁止加载其他父权重。首卡真实E20两臂forward/backward，核实L768／U256、LS一次U或NAMS六次U／共享计划／实际浅中层与头有限梯度、RDrop／AsKL0／原L信道64、Uphysical0、临时hook移除。随后在三个原E20父模型上以固定V720/seed做冻结无更新配对诊断，预定raw接受正确率与真实TX配对一致率均≥99%、两层实际应用>0，不以混合后分类精度作门槛。不复用E30的配对数值。PASS立即丢弃探针、正式重载原E20父权重，按固定六行继续；科学条件失败则在正式训练前保留证伪记录。两次同技术失败停止新增派发、保留健康进程和产物。

GPU5／6／7固定seed2026092701／02／03。LS臂等待同seed MS全部GPU预测完成，NAMS臂等待旧NAMS全部GPU预测完成且新LS已派发；旧NAMS启动未CUDA的PID与本控制器pending PID纳入容量并集。每卡最多两条本任务、含外部最多三条、空闲≥10000MiB；CPU唯一scorer排旧NAMS终态之后。环境/CWD/命令及不可覆盖发布／输出root见experiment.json：N607普通账户、固定CVS-RFFI环境、FP32／TF32off，一代码Git commit、单archive本地远端SHA对照和远端compile。不停止、重启或热改其他健康任务。

每row冻结最终学生后自动raw与同一DURA v1.0各clean＋六practical，168000条/景，六模型共84cell；全row预测固定后独立truth-last。完整报告F20/F21及两臂对照、三seed／场景／TX／RX／day、修复损伤、接受／曝光与成本；DURA只作冻结参考，不用于新F训练，G＋F20仍含G的源U权限。原raw验收为clean同时比两对照至少高1pp、六景均值严格高1.5pp，当前ACTIVE。源纯度和更长机制暴露不证明目标收益；既有测试暴露如实披露，不称新的独立盲测，无目标调参、重排或选择性重跑。预期保留smoke/E20源配对、全部新segment30/50的6660条step记录、两段hard_pl_state、最终E50 checkpoint/完整继承、source graph/native检查、双七景predictions/scores及完整报告。


## 实际三卡启动

VERIFIED：执行 commit `80fe494716b538b504a32efa9245d2e58925fc6f`，一次 archive 传输对照、远端 compile、真实原 F21 E20 两机制 smoke 及固定源 V720/seed 配对规则均 PASS。三个 E20 LS seed 已在 GPU5/6/7 实际运行，PID/CWD/完整命令/GPU及原生日志增长已独立核实；同 seed 的原 NAMS 健康任务保持，第二机制三行按预登记等待交接，每卡最多两条本任务。源诊断所有 seed 的伪标签精度及同真实 TX 配对均≥99%，源 V 无状态更新。两次 smoke 的 parent_epoch=20、parent_updates=4440；smoke 共用旧接口中的 formal_reload_original_F21_E30 仅是旧字段名，实际父权重与正式 reload 均严格为 E20，并未继承原 E30。

新方法仍累计 E50/11100，新增30轮/6660步，原生10＋20相衔接；不增加模型预算。全部正式 raw/DURA 七景会按队列自动完成后独立评分，目前无本候选目标分数。


## 两机制六条真实训练均已启动

VERIFIED：旧NAMS逐seed全部GPU预测完成后，三个E20 NAMS新worker已按约定接替，各卡LS/NAMS两条真实训练健康，PID/CWD/命令/GPU与原生step、日志增长逐条确认。LS自己的native10段2220步已闭合并进入自己的20段；不是继承原E30。逐行实际曝光、RDrop／跨RX／六前向共享style计数见source_six_worker_activation.json；目前仅源训练前缀，无本候选目标成绩。
