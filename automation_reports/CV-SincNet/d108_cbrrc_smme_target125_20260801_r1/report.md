# D108-CB-RRC-SMME/r1完整125研发与实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d108_cbrrc_smme_target125_20260801_r1`|
|日期/operator|2026-08-01；主agent负责协议、集成、数据与结果分析；Terra Max子agent分别负责DA核心、head核心和N607运行|
|目标|保留D92强表示和equal-prior LDA，以独立的support-only DA与类对称margin校准同时改善旧类floor、新类注册和K1|
|比较目标|完整125的D62、D92、SVRN-qKNN-BCRR；D91仅列15行development证据|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|本地分支`codex/stage2-da25-r1`；只提交，不push、不上传GitHub|

D107完整125中M_JOINT仅为after old=47.26%、after floor=17.87%、seen-new=30.11%、H=35.95%，显著弱于D92；D107已淘汰且不创建r2。D108不复用其signed表示、anchor-centering或kernel ridge。

## 冻结方法与四臂

候选：`D108-CB-RRC-SMME/r1`。

1. 全部臂保留D92的288维表示：ReLU z_id160＋原FFT96＋原RF32；只允许DA臂改变ReLU块。
2. CB-RRC对before旧类support逐行单位化，以六类等权方式冻结坐标能量`e_j`；零ReLU块保持零。
3. 对每条support/query独立应用`r_j(x)=sqrt((e_j+mean(e)+eps)/(u_j(x)^2+e_j+mean(e)+eps))`，先得到`unit(u(x)*(1+r(x)))`，再乘回该行输入ReLU块的原始L2范数；增益在(1,2]，不中心化、不翻转ReLU mask，也不改变D92的z_id与FFT/RF块间权重。
4. SMME保留D92 equal-prior LDA logits。对每类support计算自身logit相对其余类logsumexp的均值margin`m_c`，冻结零和偏置`delta_c=mean(m)-m_c`；query分数为`g_c(q)+delta_c`。
5. 四臂固定：`M0=D92表示＋D92头`、`M_DA=CB-RRC＋D92头`、`M_HEAD=D92表示＋SMME`、`M_JOINT=CB-RRC＋SMME`。每个row全部执行，无路由、扫描或择优运行。
6. CB-RRC与SMME在K1均活动；不fallback，不读取query truth/role/count，不用quota、qKNN、query fit/update或global reassignment。

## 可行性复核（冻结）

1. CB-RRC只从当前row的before support冻结160维能量，after沿用，符合Stage2-B状态冻结边界。
2. old/new support进入D92 LDA和SMME时公式相同；类别重命名仅同步置换状态和分数。
3. query逐样本只读取固定状态；SMME偏置在query前由support封存。
4. DA与head作用层不同：前者压缩坐标内动态范围，后者补偿低support-margin类，不是重复校准。
5. 最大持久新增状态约320B；DA单query约160维逐元素运算，head单query仅C次加法。
6. 不变更received-IQ、physical ID、receiver/TX、scenario、K、split或`p2_min_v1`，因此复用`VALIDATED_ONCE`数据，不重验。
7. 风险是singleton support的margin偏置方差和非线性压缩可能损伤易类；由完整四臂125同row结果直接证伪，不新增性能gate。
8. 判定：`FEASIBILITY_REVIEW_PASS / DESIGN_FROZEN`，直接实现。

主agent在核心初审中检出并修复一个P1：最初实现把D92最终288维特征中的ReLU子块直接重归一到1，会把原约1:4的z_id/auxiliary块权重改成约1:1。修正后CB-RRC严格保持每行输入ReLU块范数，K10实测最大绝对误差为`1.1920928955078125e-07`，23项专项测试通过；这是表示接线修正，不是方法调参。

## 冻结完整125与发布流程

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102,713103,713104,713105,713106`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|闭包|125 outer、375 scene、1500 arm pairs、3000 prediction surfaces、500 outer-arm聚合行|

精简流程固定为：两个核心模块并行实现→主agent接入已成熟的D107矩阵/不可覆盖prediction/独立truth scorer→`ssr-gpu`窄测试→一次真实checkpoint no-truth smoke→独立复核P0=0/P1=0→Git提交和不可覆盖run登记→N607完整125。不得增加source-held性能筛选、候选扫描、重复数据验证或小矩阵择优；性能弱时完整计分后立即淘汰并研发D109。

## 预登记运行面

已锁定D92 matrix SHA256=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`；Phase1 checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；D81 ground component固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`，manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`。D108不使用D106 RDCE资产。每个D92 job的authority bundle/COMMIT将逐项绑定进plan。

其余本地改动、验证命令、commit/hash、远端同步映射、精确Python/CWD/日志/PID/GPU和期望artifact将在实现冻结后补齐。N607发布由唯一实验子agent负责，默认先直连只读preflight；完整预测支持GPU0-7独立不可覆盖分片，所有分片合并验证后才封存3000个surface并开放truth。停止只允许P0协议/安全故障或至少两个不同row出现相同确定性零prediction异常指纹，绝不因中间性能停止。

## 本地实现、验证与发布封存

|项目|证据|
|---|---|
|DA与head核心|commit=`eb229847`；CB-RRC23项、SMME8项测试通过|
|D92四臂核心|commit=`ada3fcc4`；8项测试通过；M0逐值保持D92正式评分|
|矩阵与truth证据面|commit=`6d2a415a`；Target125与truth各4项测试通过|
|真实runner|commit=`9d6b0f65`，文件尾修正=`534648ae`；真实D92 sealed runtime→288维特征、四臂pair/score、不可变smoke预测、8个modulo分片和严格合并|
|本地联合验证|`ssr-gpu`下核心与证据面`47 passed`，runner`6 passed`；`py_compile`、CLI帮助、非法shard负例和`git diff --check`通过|
|独立复核|核心/证据面`P0=0,P1=0`；runner`P0=0,P1=0 / RUNNER_REVIEW_GO`|
|method lock|`configs/stage2_d108_cbrrc_smme_r1.json`；SHA256=`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`|
|发布源码|commit=`534648ae5c9f72e0bbbfced73846197442ffaa9b`；archive=`E:\type10-7\code\snapshots\d108_cbrrc_smme_target125_20260801_r1_source_534648ae.tar`；SHA256=`eb77728f5dc89167164bbed2f5e96058c47d5a18d39d23e90f323b7d48a9a802`|

远端不可覆盖run root预登记为`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target125_20260801_r1`，源码目录为其下`source`，Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD固定为源码目录。D92 output root为`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`；checkpoint为`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。prepare固定写入`prepared`；smoke固定写入`smoke`；8个prediction shard分别写入`shards/shard_0`至`shards/shard_7`；合并写入`predictions`；truth与score分别写入`truth_catalog.json`和`score`，全部不可覆盖。

发布子agent仅执行以下冻结链路：fresh direct preflight→同步并校验archive SHA→远端解包和D108文件`py_compile`→prepare→GPU0真实row0/clear无truth smoke→比较M0 before/after的query ID与预测和D92参考完全一致→GPU0—7各运行一个固定shard→严格合并125/3000→prediction封存后build-truth与score→回收artifact并退出所有SSH连接。每个shard命令固定使用`run_d108_cbrrc_smme_target125.py predict-shard`、相同plan/context SHA、`--shard-index 0..7`和对应`--device cuda:0..7`；不得按局部性能停止、重启、调参或选择性补跑。

期望artifact：`prepared/target125_plan.json`、`prepared/target125_context.json`、`smoke/smoke_receipt.json`、`smoke/smoke_predictions.json`、8个`prediction_shard_manifest.json`、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`、每阶段日志/PID/exit与GPU/process核验记录。系统性技术停止条件仅为P0协议/安全故障，或至少两个不同outer row在生成prediction前出现相同确定性异常指纹；性能值不得触发停止。

## N607 r1技术闭包

2026-08-01 direct preflight通过，8张RTX3090均为空闲，runroot确认不存在后首次创建。archive远端SHA256与预登记完全一致；解包源码的D108文件`py_compile`通过。Git archive对JSON执行LF规范化，archive内method lock SHA256为`0ed48795…`；发布子agent按明确授权精确同步本地已验证字节后，远端method lock SHA256恢复为`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`，JSON解析通过。

prepare成功封存125outer、375scene、1500arm pair和3000预测面；plan文件SHA256=`ee9318e593ebd532511173478b9d2637b306bcf4cab48b8c5fff21e2f90c9a46`，context文件SHA256=`569f9343587be5e107aa27aebf841bdbc29d8704d9feac40d280499aee82f0f8`。GPU0的row0/scene0真实no-truth smoke运行约20秒后，在首个pair构建、任何prediction生成前以exit=1 fail-closed：`D42 sklearn runtime version drift: expected 1.7.2, got 1.7.0`。

r1实际prediction surface=`0`；8个shard、merge、truth和score均未启动，严禁形成性能结论。事后D108进程=`0`、GPU compute进程=`0`、8卡均`0%/1MiB`，本机`ssh.exe`与TCP22连接均清空。服务器与本地partial artifact完整保留；本地证据位于`artifacts/remote_r1`。N607没有现成sklearn1.7.2环境：`CVS-RFFI`为1.7.0，`SDG-SEI`为1.3.2，`ssr-gpu`不存在；未安装或升级任何包。后续只能在本地完成已有D81式1.7.0/1.7.2严格兼容修复、测试、独立复核和新commit后，以新run ID执行，不得恢复或覆盖r1。

## 性能口径

主表使用125个outer-row的before old、after old、before floor、after floor、seen-new、H和forgetting均值；post correct使用全量正确数，并与D62/D92/SVRN同口径配对。D91继续标注为15行development，不进入正式125排名。完成后补充arm、slice、receiver、scene、逐类和异常表，所有指标保持同一row/arm绑定。
