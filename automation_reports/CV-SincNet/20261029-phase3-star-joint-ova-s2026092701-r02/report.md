# PAMF第三阶段：开放损失联合表征与特征保真

## 预登记（2026-10-30执行准备，保留20261029候选ID）

Run `20261029-phase3-star-joint-ova-s2026092701-r02`；code `5aa041122396d48c49b8cb3d5912649bdc3217e1`；两组detach_control/joint，唯一主joint::ova_max。方法/门槛见`docs/PAMF_PHASE3_STAR_JOINT_OVA_PLAN_20261029.md`及`configs/pamf_phase3_star_joint_ova_20261029.json`，在本次真实源IQ前固定。

两组均strict原F0 E50/11100/232993/default6head初始化，CE_L+KD_L(T2)+5unitfeatureteacher保真+1辅助OVA；区别仅OVA到F的梯度detach/开启。原defaulthead随F联合更新，aux独立160->6 binary966参数，只拒识不替身份。L128uniform/OE128 domainuniform一次concat256前向，无随机增强/PL/EMA/VICReg；每组F2220＋aux2220更新，F累计13320，每组4440 optimizer.step，两组8880调用，不能把它当8880F更新或十遍自然epoch。model eval withgrad、BN/dropout冻结。

本r02仅替代r01的技术停止，不继承其模型：r01 teacher完成、两train在任何smoke/更新前将完整900722参数wrapper误与232993部署encoder计数比较而退出。现计数对象改为id_backbone.encoder，完整wrapper/strictstate/defaulthead/optimizer集合保持。真实native graph回归RED→PASS，错误encoder仍拒绝，且唯一针对该P1的独立复核PASS；不重复全候选审查。本r02全新run/release/teacher，完整重导原F0教师，科学矩阵、预算、种子不变。r01产物保留，无性能分数。

真实输入仅cached G10 P1。teacher只由原F0同输入L6300/OE6134导出unitfused160；L全6teacher logits复用cache。V27000仅每组三score×六候选预测类τ；known42组C90/原F0同P1closed−open2pp。原F0baseline缓存不被candidateclosed重置，candidate闭集本身不合格时acceptall回退仍科学fail并完整novel预测。所有NN/18τ先冻再source novel3150；原F0同P1 novel额外3150feature参考/row，只供独立source跨RX1NN诊断，不拟合novel、不冒充AN/注册。

source五RX1/3/4/6/8、三日2021-03-08/15/23，与训练RX相同，保留三novel历史曝光限制；没有targetU/Q/正式RX24-18句柄。source L标签/OE成员监督及原F0 L6300/U56700/V27000祖先如实披露，全系统不是零源标签。G10 MIX01来自既有星地received/DURA缓存，真实raw/信道重生/Gforward均0。合成G10 P0 checkpoint probes10行单列(teacher2/train4/eval4)，不充作真实P0或P1/G管线smoke。预期总F前向1215684，重复曝光不增42584独立原物理记录。

晋级只看joint::ova_max：known42保护、unknown总体/5RX/6scene≥90%必要覆盖完整、source novel crossRX1NN较original最多下降2pp。stableopaqueID打破cosine ties、排除同RX，trueTX只independent scorer。source通过最多准备未来新run targetfalsification，不是target成功；失败NO_PROMOTION，control/energy/routed不换主，AN/HOW=N/A，完整Phase3 ACTIVE。

### 精确执行与边界

- Host N607；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-joint-ova-s2026092701-r02-release`；唯一owner `codex-phase3-star-joint-ova-20261029`。
- 命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_joint_ova_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-joint-ova-s2026092701-r02-release/runtime.json`。
- source cache `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`，teacher output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-joint-ova-s2026092701-r02/teacher`，row outputs `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-joint-ova-s2026092701-r02/rows/{detach_control,joint}/{train,predict,score}`；logs `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-joint-ova-s2026092701-r02/logs`；configs `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-joint-ova-s2026092701-r02/configs`；local `E:/type10-7/local_artifacts/pamf_phase3_star_joint_ova_20261029_r02`。
- 原checkpoint `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/final_ssdg.pth`；contract `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/source_contract.json`；原Phase1 sourcecache `/home/szu2070436088/2510044040/CV-SincNet/runs/20261023-phase1-practical-residual-full2x2-m600-r01/data/source/F0`。严格parent身份及现sourcecontract392005匹配，不加载任何按Phase3结果挑选的增量。
- 新teacher/train config含L/OE handles，无V/novel/target；eval才开放cache V与freeze后novel；private source truth仅CPU scorer `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`。
- GPU teacher串行，随后max2 rows在不同实际free≥4096MB卡运行，每卡≤2新增。每row train→eval→独立CPUscore即时衔接，失败row保留、独立健康row继续。其他owner工作不改。teacher技术失败不启动rows，科学失败不停止/热改/重启。
- 预期teacher complete/smoke/L+OEunits；row train final.pt/complete/training/2220steps/smoke，predict three-score/logits/units/18τ/calibration/complete，score results/complete；launch/pipeline/rowstate/完整日志及观察。

验证：原候选29集成tests/45subtests、launcher字段修复新增1回归PASS，原unique30/45sub保留；本次只新增1个参数对象回归，修复后1PASS3.58s，r02五AST/runtime schema PASS，未重跑未变suite。原唯一候选独立P0/P1 PASS，本参数P1一次定向复核PASS，未加第二次全审。一次本r02 N607 preflight检查原source契约/缓存四role/真实显存通过、target0；release一次archiveSHA比较/remotecompile、first-step actualcheckpoint noquerysmoke、启动PID/CWD/cmdline/GPU/log增长和truth-last同row分析按原最小流程执行。

技术停止限祖先/角色错误、输出碰撞、非有限或无法完成合法prediction，保留partial、不原地重启。无performance结果或完成声明，目标仍ACTIVE。
