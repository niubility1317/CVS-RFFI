# 旧 SIG_TIME 叠加 RDrop：四个未启动 E50 行的技术恢复

状态 RUNNING，四行均已启动；真实 checkpoint 无查询冒烟、进程身份和日志增长独立核验 VERIFIED。原E20/E50×三条件×三seed为18行，原run已实际启动14行。Popen后瞬时/proc读取导致原调度器退出；一个未记录worker实际仍健康。CPU-only128次复现2次GPU字段暂缺后同PID恢复。保留failure、所有输出和健康任务，不原地重启或重训已有14行。

新run仅训练RDROP seed02，以及ORIGINAL/TWO_CE/RDROP seed03，共四个未启动行。每行仍50轮/11100更新/33L+17U，RAW_HZ、普通增强、LS.01、默认头joint、原U/ownEMA、固定final全部保持。metadata仅run_id/output_root/owner变化；实际模型/训练/预测/评分继续原不可变release，模型代码commit为b3ac442e2d33a5b315fb3251c30e0ad7083566bc。新调度代码commit记录在发行包release_commit.txt。八个配置负例、逐字段相等和八个启动回归PASS；修复前同回归按原错误失败。无新参数、换头、另训头、旧CP继承。

复用原首seed E50源资格PASS；preflight只读冻结源metadata复算，不读target指标。四行由不存在的output/process确定，保持同源6300L/56700隐藏U/27000V和VALIDATED_ONCE句柄，不重验数据。原14+新4仍630model-epoch/139860更新；新4成本200epoch/44400更新/4262400额外L前向/3855600次U观察/672000clean决定/72分层记录，另8个ownCPsmoke合成包。

先保存已启动PID，再在2秒内只等同PID的CWD/argv/GPU可见，不重复Popen。超时保留归属/输出，停pending、不杀健康子进程；修复后同指纹复发不盲目重发。共享原receiver_residual_capacity4.lock，all8空闲优先、all-owner含preCUDA最多3/GPU、free≥12000MiB，CPU2、loader0。预计3348MiB/worker，启动后测量。

新dispatcher观察原RDROP seed01/ORIGINAL seed02/TWO_CE seed02的缺评分闭环。已完成评分复用，健康训练自然结束，固定prediction后才用原命令fresh CPU CUDAempty评分，不重训、不覆盖partial score。RDROP seed01缺失评分已独立恢复，其原预测不变。每新row冻结后立即clean168000预测与独立truth-last。最终合并14旧+4新，保留两个root，不伪装本4行为完整三seed统计。

ordinary N607 CVS-RFFI；CWD为PROJECT/releases/cvs_originalSIGTIME_rdrop_e50_technical_20261015_r02；命令为CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_recovery/runner.py experiments/cvs_sigtime_rdrop_recovery/configs/matrix.json；owner为codex/root/original-SIGTIME-RDrop-technical-20261015；唯一新output root为PROJECT/runs/20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02。原root只闭合既有预测的缺评分，旧failure不删除。全部逐row配置/命令/路径/seed/数据权限/停止规则唯一维护在experiment.json。

一次聚焦P0/P1技术修复审查已PASS（review_identity_tangent，未发现P0/P1），尚需Git提交推送与OID读回、一次preflight/archiveSHA/remotecompile及启动后独立PID/CWD/argv/GPU/log核验。每新worker首先保存/严格重载own scratch CP并对2个合成包无query smoke，PASS立即训练。未发布或未评分不宣称收益。

E20已有ORIGINAL83.115675%、TWO_CE82.486111%、RDROP83.942857%，只是报告事实，不决定本恢复；固定主条件仍E50_RDROP，84%与创新训练目标未声明完成。资源协调chat回写返回no rollout，不阻止实时共享锁/all-owner容量核对。额外设计审批、无关全套测试、重复科学审查与seal/hash链均REJECTED_EXTRA_GATE。


恢复代码 33f6171a0fec6ac432df672d0c6f542b8d71434c 已提交推送，远程 OID 独立读回一致；实际模型代码仍为 b3ac442e2d33a5b315fb3251c30e0ad7083566bc。一次 N607 preflight、单发行包 SHA 比较与远程编译 VERIFIED。四行在 GPU0/2/7 运行，快照 all-owner 均不超过3进程/GPU，实测 3346 至 3350 MiB/worker，CPU2/loader0。全部默认头联合训练、真实源角色和原配方均已核实。原14行已全部自然完成 clean 预测与独立评分，无旧模型重训或停止；原 failure 保留。新4行冻结后自动逐行预测和独立 CPU truth-last，完整18行统计待齐。详见启动证据 launch_verification.json。
