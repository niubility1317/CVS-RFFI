# D108-CB-RRC-SMME/r1完整125研发与实验报告

状态：`DESIGN_FROZEN / IMPLEMENTING`

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

## 性能口径

主表使用125个outer-row的before old、after old、before floor、after floor、seen-new、H和forgetting均值；post correct使用全量正确数，并与D62/D92/SVRN同口径配对。D91继续标注为15行development，不进入正式125排名。完成后补充arm、slice、receiver、scene、逐类和异常表，所有指标保持同一row/arm绑定。
