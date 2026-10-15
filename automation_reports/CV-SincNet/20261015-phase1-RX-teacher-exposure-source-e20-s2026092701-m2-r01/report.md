# 两教师 E20 源域互补性小试

状态 **PLANNED，已登记，尚未发布/训练**。固定seed2026092701，SAMPLE_80 / EXCLUDE_RX1 两个原SIG_TIME/CENTER25K/defaulthead从零纯CE E20模型，各5040合法L、每TX/day280、4440更新/568320主L前向；普通组各TX/RX/day56个L，排除组4RX各70。默认头正常joint，原AdamW lr.0002/wd.0001/cos，LS0/native dropout，无U迭代/EMA/增强/辅助损失/新参数/换头。实际每批L base_index/TX/RX/day与固定来源核对。原source载体6300/56700/27000 VALIDATED_ONCE复用，先筛L/V及弃U，再构造模型；有效roles5040/0/21600独立保存，不冒充full-source父来源。

两组fit/calibrationV均RX3/4/6/8=21600；RX1源V5400仅未materialize的dataset引用，fixedfinal20冻结后才做默认头预测（六TX×三day×300）。比较同ID错误/meanCE/真类概率、修正−破坏与day/TX/类别对。独立CPU仅在两个来源证明和源预测完整后，加载原全源BASE_SSL seed01 E20/LS0的固定student+ownEMA作源V原门控回放，不进入教师训练。weak为原clean U输入，strong为native std.015且student dropout激活，teacher eval；固定sourceV shuffle2026101502/view2026101503/batch256；gate只获物理metadata，不获TX标签。全部model参数/buffer/grad不变；不拟合阈值/分类头、不读取U IQ/TX或target。它是final A0的sourceV proxy，不能推断实际U精度或早期teacher行为。

唯一科学条件：两组final训练LCE≤.5，EXCLUDE全5400错误严格更少且meanCE不高于SAMPLE，并且固定A0拒绝区域非空、纠正严格多于破坏。任一不通过即收束固定配方，不扫描阈值/样本/epoch/seed救援。通过也不自动扩五教师库、学生或target；下一步另登记。全/接受/拒绝/day/TX/类别对风险完整保留。RDrop12固定E20/E50矩阵继续原计划，本项参数在未读取其target分数时固定。另一owner只有E50/4464L/720dev草稿，本项不是其已完成结果或对照，也不干预其独立科学范围。

ordinary N607 CVS-RFFI，CWD `PROJECT/releases/cvs_RX_teacher_exposure_source20_20261015_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_rx_teacher20/runner.py experiments/cvs_rx_teacher20/configs/matrix.json`；owner `codex/root/RX-teacher-exposure-source20-20261015`；唯一root `PROJECT/runs/20261015-phase1-RX-teacher-exposure-source-e20-s2026092701-m2-r01`。2个已固定就绪worker按all8 idlefirst，共同 `receiver_residual_capacity4.lock` 内读全部actual/preCUDA owner、cap3/GPU/free≥12000MiB、选卡/Popen/记录。CPU2/worker、loader0，计划3.3GiB/worker，实际显存/时间另测。CPUdispatcher与freshcompare CUDA为空。预期40modelepoch/8880更新/1136640Ltrain、10800teacherheldV前向、10800A0proxy前向，合成smoke另计；native训练每轮fitV诊断及固定A0历史训练成本分别披露。

技术停止：非法角色/来源或U迭代、非有限/预算/预测错误、已有输出碰撞、worker或CPUcompare失败保留所有产物，停止未启动行，已启动健康行正常结束，无自动retry/kill其他owner。每launcher第一步保存/重载own初始CP，2个合成无query前向PASS即继续。源码/配置actualGitcommit、一次聚焦负例及actualmodel检查、一次独立下一运行P0/P1、一次N607 preflight/archiveSHA比较/remotecompile与独立PID/CWD/argv/GPU/log增长按最小流程执行。预期初末CP、实际源subset/role/args、完整step/epoch、teacher heldV默认概率、A0gateproxy、matrix资格与报告均在experiment.json。source-only明确不测试target/satellite。

当前44个异常负例及4个资格否决边界PASS，2真实模型200state初值一致、native222 dummy pairs从不迭代U、128合成L的t1/t2/SIG/W梯度有限非零，审计hook与原CE梯度逐位一致。首次新增fixture因Windows路径表示差异失败已修正，未涉及真实IQ/CP/训练；一次未归档的无用__init__引用已移除。此前prototype的21项结果保留为实现历史。本地检查不等同实际教师互补或泛化收益。

背景：[用户路线落实说明](../../../analysis/cvs_rx_teacher_exposure_source20_20261015.md)。互补域教师有[DomED先例](https://proceedings.mlr.press/v306/song26af.html)，不宣称全球首创。84%与完整训练机制研究目标仍active。REJECTED_EXTRA_GATE：额外设计/spec审批、report-only重复review、额外哈希/receipt链和新调度平台不进入本流程。
