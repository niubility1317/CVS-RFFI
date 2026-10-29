# PAMF第三阶段：源样本局部支持与跨接收机共识

## 预登记（2026-10-30）

Run `20261030-phase3-star-local-support-s2026092701-r01`，实际code `84cdc012f7d9641b283c407f6048bfe4e6e21cf6`。一个冻结原F0、三分数pooled_known/pooled_relative/rx_consensus_relative，唯一主最后一项。方法/数据/接口见`docs/PAMF_PHASE3_STAR_LOCAL_SUPPORT_PLAN_20261030.md`和`configs/pamf_phase3_star_local_support_20261030.json`；参数k8/eps1e-6事前固定，不扫k、不按novel换主。

原Phase1三视图F0 E50/11100/default6head/232993固定，祖先L6300/U56700/V27000、sourcecontract392005、训练标签比例10%。bank复用jointOVA r02 teacher原F0同P1 L6300/OE6134单位fused160/code5aa041122396d48c49b8cb3d5912649bdc3217e1，它不是r02增量权重。teacher/cache/actualreader的原F0和DURA身份一致后才使用，L/OE IDs及RX同role顺序，perTX或OE真实TX/K不用于bank评分。

三分数：负pooled known平均8邻距；pooled log(OE邻距+eps)−log(known邻距+eps)；每source RX分别取8邻距后dK/dOE各median五RX再取同log比。仅source bank分组，不接收query RX/场景/标签。FP64逐条unit/距离clip(2−2cos,0,4)，128查询分块，bank在V前固定。所有身份仍为原默认headargmax，不能按邻居身份替代。

G10 MIX01既有P1真实输入，不重生received/不新G。L/OE bank特征复用，无新L/OE neuralforward；V27000原F0 P1新特征仅用于三分数各六预测类τ，trueTX宏known42 CK90/originalF0closed−AK2pp保护与complete ties不变。baseline仍原Vcache[:,P1]，fresh闭集另列，不因数值差异重设baseline。全部bank/model/18τ固定后才novel3150新原F0 P1特征/默认头预测，CPU独立truth-last在本row prediction完成后连接私有TX。总F27000+3150+2syntheticG10P0smoke=30152，真实G0、NN/optimizer更新0，无L/Us/targetUQ/formal读取；合成P0不是正式P0/P1或G管线测试。

42584原物理记录仍同五sourceRX1/3/4/6/8、三日2021-03-08/15/23、历史novel19-15/19-16/7-1各1050，非新RX/新TX盲测。V标签是额外source校准监督，OE成员关系也是已用监督，不称全系统零源标签。原unknown/角色/K/配额只scorer可见，不进入fit。

### 执行

- Host N607；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-local-support-s2026092701-r01-release`；唯一owner `codex-phase3-star-local-support-20261030`。
- 命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_local_support_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-local-support-s2026092701-r01-release/runtime.json`。
- output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-local-support-s2026092701-r01/rows/original_F0/{predict,score}`，configs/logs/runstate各独占；local `E:/type10-7/local_artifacts/pamf_phase3_star_local_support_20261030_r01`。
- originalF0 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/final_ssdg.pth`，sourcecontract `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/source_contract.json`；teacher `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-joint-ova-s2026092701-r02/teacher`；G10cache `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`。使用VALIDATED_ONCE角色句柄，不新增raw或数据重验。
- actualcheckpoint first-step合成星地smoke先于新source消费，PASS立即继续。一个GPU source预测进程、实际free>=4096MB，每卡<=2新增，不停止或修改其他owner。source完成即独立CPUscore，private source truth仅该进程 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`。
- 预期source_bank六fields、known/novel单位features/defaultlogits/rawscore/margin、18τ/calcomplete/smoke/rootcomplete；score results/complete；全部configs/logs/launch/pipeline/rowstate/observations。源码禁止bank/query统计更新，source config不含truth/target字段。

主报trueTX宏AK/RU/H及CK、原baseline/freshclosed/配对差、known42/unknown12/逐RXsceneTX/AUROC/OSCR/CI。只有主known42及unknown总体/五RX/六scene宏RU>=90同时过才准备另记小target证伪；否则NO_PROMOTION，不换控制/重新校准或扩seed。source通过不是完整跨新RX/星地无标签适应/未来匿名注册达成，AN/HOW=N/A、完整目标ACTIVE。

NN部署参数F232993+G266951=499944，无aux；源bankFP32约7.96MB、FP64工作约15.92MB，RX/IDs/workspace另计，不能只报NN参数掩盖存储。生产CPUreadout_seconds只计V/novel核心score调用，与F/校准/独立复核分开，非B1部署时延；不预先承诺加速。

技术stop仅来源/角色/权限错误、非有限、输出碰撞或无法完成合法prediction，保留全部partial，不因性能低停止或重启。未知启动结果先读回，不盲目重复提交。没有其他owner干预权限。

验证：20tests/46subtests PASS6.11s、四AST/实际runtime schema PASS；一次独立P0/P1 PASS，定向7tests16sub与主suite重叠不加计。覆盖手算距离/ties/RXmedian/查询行独立与bank不变/父数据角色/默认头/原cachebaseline/true-label-before-validation/不可覆盖与实际终态字段。一次本run N607 preflight核实原F0 source契约／teacher身份／四role句柄／GPU剩余显存PASS，target0。单code/config commit、一个archiveSHA比较/remotecompile、actualcheckpoint smoke、启动PID/CWD/cmd/GPU/log及全部预测后的独立评分，未添加报告审批/封印/hash链或重复父suite。
