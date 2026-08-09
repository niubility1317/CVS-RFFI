# Phase1 CAGM后冻结42步v2预登记报告

状态：`ANALYZED / TECHNICAL_BINDING_PASS / REJECT_P1_CAGM_PERMANENT / NO_PHASE3_PROMOTION`

## Runner阶段记录（2026-08-10，postfreeze Revision2独立执行链）

- direct `n607_ssh_preflight.ps1`通过且`preflight_exit=0`；本地SSH进程/TCP22均为0。N607项目根可见，GPU0–6各约1MiB，GPU7约498MiB为既有SCB v4；未干预SCB。
- 不可变release六个postfreeze成员SHA、ManySig SHA与训练v2的12个final checkpoint SHA全部匹配；training v2 receipt/status合同已核到schema v2、C/G joint-mask/AdamW约束，training进程为0。
- postfreeze root/log/outer启动前均确认ABSENT；v1/v2训练路径保留不改。远端`py_compile`（5个postfreeze Python文件）、4个脚本`--help`、launcher`bash -n`均通过；`--dry-run`严格42条：`CAGM_CLEAN_EXPORT=12`、`CAGM_LEO_EXPORT_AND_BIND=12`、`FROZEN_LOGITS_PROXY_BINDING=12`、`CAGM_PAIR_SCORE=6`，无路径创建或性能读取。
- 第4节exact命令调用次数=`1`；SSH约49秒超时后仅清理本地SSH PID=`28260`并确认TCP22清零，未重发。远端已落地并完成；wrapper/launcher在确认前已退出，12个candidate PID及GPU由`candidate_pids.tsv`持久记录，CWD统一为不可变release/code。
- 完整42步计数：12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON、12 proxy CSV、6 pair JSON；18阶段日志（12 candidate＋6 pair）和1份PID表，outer存在但大小为0。F6 pair JSON存在`matrix_aggregate`技术字段。
- 6/6 pair JSON技术binding通过：postfreeze root、matrix ID、training root、C/G checkpoint binding、common training binding、receipt revalidation与technical binding均通过；C/G joint-mask与AdamW/initial-state合同保持训练v2证据。18阶段日志中Traceback、RuntimeError、CUDA OOM、argparse、协议/技术失败指纹计数均为0。

### 42步最终技术交接表（不读取或解释性能）

|candidate|GPU|child PID|CWD|5项candidate工件|binding技术结论|
|---|---:|---:|---|---|---|
|F1C_CAGM12|0|724341|immutable release/code|clean/LEO/binding/proxy JSON+CSV=5/5|PASS|
|F1G_CAGM12|1|724343|immutable release/code|5/5|PASS|
|F2C_CAGM12|2|724346|immutable release/code|5/5|PASS|
|F2G_CAGM12|3|724349|immutable release/code|5/5|PASS|
|F3C_CAGM12|4|724353|immutable release/code|5/5|PASS|
|F3G_CAGM12|5|724356|immutable release/code|5/5|PASS|
|F4C_CAGM12|6|724357|immutable release/code|5/5|PASS|
|F4G_CAGM12|7|724361|immutable release/code|5/5|PASS|
|F5C_CAGM12|1|724344|immutable release/code|5/5|PASS|
|F5G_CAGM12|0|724342|immutable release/code|5/5|PASS|
|F6C_CAGM12|3|724351|immutable release/code|5/5|PASS|
|F6G_CAGM12|2|724347|immutable release/code|5/5|PASS|

|pair|pair JSON|root/matrix|C/G binding|common/receipt|technical binding|
|---|---|---|---|---|---|
|F1_C_vs_G|present|PASS|PASS|PASS|PASS|
|F2_C_vs_G|present|PASS|PASS|PASS|PASS|
|F3_C_vs_G|present|PASS|PASS|PASS|PASS|
|F4_C_vs_G|present|PASS|PASS|PASS|PASS|
|F5_C_vs_G|present|PASS|PASS|PASS|PASS|
|F6_C_vs_G|present＋matrix_aggregate|PASS|PASS|PASS|PASS|

### 小工件bundle、manifest与清理

- 远端bundle=`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm_postfreeze_20260810_v2/phase1_cagm_postfreeze_20260810_v2_small_artifacts.tar`；本地=`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm_postfreeze_20260810_v2\artifacts\returned_small\phase1_cagm_postfreeze_20260810_v2_small_artifacts.tar`。
- bundle大小=`28733440`字节，SHA256=`4453ec5f6eeabb801a83152fd68c2da277306d56856a37ca3a13ed4e80398762`，members=`63`（62个小工件＋manifest），`.pth/.npz=0`。
- 本地manifest=`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm_postfreeze_20260810_v2\artifacts\returned_small\phase1_cagm_postfreeze_20260810_v2_small_artifacts.manifest.txt`，大小=`8955`字节，62条，SHA256=`4d903062a9845c3274fd8c1ff2108b9056912af753ed307096b938f04fda6967`。
- 远端目标postfreeze进程已为0；GPU0–6回到约1MiB，GPU7仅既有SCB v4约498MiB；本地SSH/TCP22均为0。未下载`.pth`或`.npz`；pair JSON与小工件已准备给主代理分析。
- 本地bundle验证的隔离提取目录`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm_postfreeze_20260810_v2\artifacts\returned_small\_manifest_extract_tmp2`仍保留；其仅含manifest临时副本，不在bundle、不含`.pth/.npz`，不影响远端或实验工件。

日期：2026-08-10

## 1.目标与冻结评价合同

|字段|冻结值|
|---|---|
|postfreeze run ID|`phase1_cagm_postfreeze_20260810_v2`|
|training run|`phase1_cagm12_20260810_v2`|
|目标|对6个same-fold C/G pair执行12 clean＋12 LEO/binding＋12 proxy＋6 pair共42步，并形成一次性非补偿判定|
|实现commit|`0ba9675e6fff859aea78319941ab68335c744cc9`|
|训练证据commit|`87ac2251539bf91a6a2aa4566f515de206964eac`|
|独立复核|训练与postfreeze Revision2：`P0=0、P1=0、ALLOW`|
|候选终态|完整门任一失败=`REJECT_P1_CAGM_PERMANENT`；全部通过才可进入主代理晋级复核|

本评价合同与已签字ICMT v2使用同一公平尺度，但不复用ICMT训练机制或结论：

- 仅由clean L的`z_id=feat_joint`拟合float64 totalized-L2对角Gaussian；V/proxy零fit；
- 正范数映射`z/||z||`，精确零向量映射0，nonfinite失败；ddof=1、4类等权pool、`.9/.1`shrink、`1e-6`floor、完整NLL和stable logsumexp连续`u`；
- proxy固定days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX上限400、总数400；
- LEO固定三场景、ManySig SHA、source selection、physical key、逐场景TX/RX/day；
- current pair和F6从原始checkpoint/NPZ/binding/JSON/CSV重算，不信prior派生delta；
- clean6/6和LEO18/18四项floor均不得低于C−2pp；每fold三场景overall和全18格overall不得为负；每fold`ΔAUROC>0`且`Δ(mean u_proxy−mean u_V)>0`，必须6/6，均值不可补偿。

## 2.代码、归档与本地验证

复用已在N607逐字验证的不可变release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e`。完整归档SHA256=`2cb14fc5689c9de1fd450edf3286508d016ebbcb8d6712f2c7733f88f7767e44`、`260669440`字节、`4912`members、`code/code=0`。

|文件|SHA256|
|---|---|
|`analysis/phase1_cagm_postfreeze_design_20260810.md`|`0e9c3195196c8a0021b3579afd103ce697fdac3c61deef21dfa3d68ec67ef51f`|
|`code/export_phase1_cagm_features.py`|`8b7013e777f5568bf1580f7bd2697d66bfaa17532993bd9cdcd32747003728ea`|
|`code/export_phase1_cagm_leo_features.py`|`c3919fe6b60e86fcf20984868191173e1f852b7e7b0992d3793f2ffc88afa9c8`|
|`code/evaluate_phase1_cagm_postfreeze_pair.py`|`ef9c5fd7cf89ac9827659367334e8eadeaf9e9521caa67f93aff3ddee0631ea0`|
|`code/scripts/launch_phase1_cagm_postfreeze_20260810.sh`|`9d9606b07a5037a08d48467de2d20a3d8f0edbb2393058b6f7bd6249f02e5e73`；mode100755|
|`code/tests/test_phase1_cagm_postfreeze.py`|`4a00887bda8176f3660f914dd195e03d96322bbf1d63c816973aeec7943f5efb`|

本地`ssr-gpu`：postfreeze focused`38 passed`、ICMT模板`31 passed`，联合`69 passed`；`py_compile`、`bash -n`、`git diff --check`通过；dry-run严格`42=12+12+12+6`。反例覆盖旧schema、joint-mask缺失/False、optimizer漂移、C/G sequence/rows/scenario漂移、400 proxy同步缩行、LEO数据替换、单场景缺day、prior summary/SHA/F6原始工件篡改。

## 3.冻结训练输入

训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm12_20260810_v2`。12臂均为receipt schema v2、`NON_PROMOTABLE_P0_DISABLED/exit8`；C6/6 control pass，G6/6 joint-mask true、AdamW、initial state empty/SHA、gradient audit与terminal pass。

|candidate|final checkpoint SHA256|
|---|---|
|F1C_CAGM12|`c2c1142135015ac362d0338af59b60195181682ecae9aca0a2c391b68415f4cb`|
|F1G_CAGM12|`8e209eae17f614c4aa564c7f0268e8543d68c0c9a6c28c1db62bfcac670b352a`|
|F2C_CAGM12|`80eebbf6462b40b4e6084eb09d3a74d0449cd13972b2e786c90bc3f50f99ecb1`|
|F2G_CAGM12|`5b588bc9b8314690990c8e69c9d1b747fe8cf3efd3d1a0c19f11ec0263c4ea52`|
|F3C_CAGM12|`92c7881db019df8f837213135f2e80fb593d08e010dadec857df635e169014b7`|
|F3G_CAGM12|`6ee3072f6785d9b7c54eb1bcce19110bb935fbad5a56ad2761f8740a8939d81c`|
|F4C_CAGM12|`875574ed683207df5c6aadab1e17293b3a89bdc99cec1815903a711b934db0a9`|
|F4G_CAGM12|`e62a3b02885fb394433c63f7a00dddd9507bace80f600bab66c62f8c8fdc2f61`|
|F5C_CAGM12|`6d28eb0d0e81c4c5df95a4d8b606a106bd111b437075e86327ceacac72b2ffb6`|
|F5G_CAGM12|`1d51e86d23319bb765f4a01916d345de04b1b28a25c31d258015f9f6cbff6a8b`|
|F6C_CAGM12|`d1fc8c35c4455835e0b30d71ec909e224e8105d45b799617e40ac7f058de13a7`|
|F6G_CAGM12|`56ddec04a3f6fe48a69fe986e8b6b3b544468036fded9edb69dabe87c555787b`|

ManySig SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。

## 4.路径、矩阵与唯一启动

|字段|冻结值|
|---|---|
|CWD|`<immutable release>/code`|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm_postfreeze_20260810_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm_postfreeze_20260810_v2`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm_postfreeze_20260810_v2_launcher.out`|
|GPU|0:F1C/F5G；1:F1G/F5C；2:F2C/F6G；3:F2G/F6C；4:F3C；5:F3G；6:F4C；7:F4G|
|retry|NO；调用异常先只读确认landed，禁止重发|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code && nohup env POSTFREEZE_RUN_ID=phase1_cagm_postfreeze_20260810_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm12_20260810_v2 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm_postfreeze_20260810_v2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm_postfreeze_20260810_v2 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v2_0ba9675e/code/scripts/launch_phase1_cagm_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm_postfreeze_20260810_v2_launcher.out 2>&1 < /dev/null &
```

## 5.健康停止、工件与分析边界

唯一Runner启动前只做一次必要的release/member、ManySig、12 checkpoint SHA、不可覆盖root和GPU核验；复用已验证release，不重新同步或修改远端代码。唯一命令后记录wrapper/launcher/12 candidate PID、CWD/cmdline、GPU和日志。仅路径/hash/覆盖、协议/receipt、Traceback/OOM/CUDA、nonfinite、两个distinct candidate同一确定性异常或缺必要工件可技术停止；绝不按性能停止，NO RETRY。

完整预期：12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON＋12 CSV、6 pair JSON、18日志与PID表。Runner在42步完整闭合前不读取或解释性能；闭合后只回收小JSON/CSV/log/binding与manifest，不下载checkpoint或NPZ。主代理随后从6个pair JSON和F6 aggregate读取完整性能、逐fold/逐场景非补偿门并更新本报告。

## 6.主控完整性能分析

主控仅在Runner完成42步、6/6 pair技术绑定通过且目标进程归零后，读取同一postfreeze v2 run的6份pair JSON与F6 `matrix_aggregate`。全部C/G比较均为same-fold、同训练根、同数据顺序、同场景与同后冻结合同；分类差值单位为pp。连续`u`只用于冻结source-proxy诊断，不代表真实unknown、FAR或Phase3能力。

### 6.1clean同折四地板

|fold|overall C→G|min-class C→G|min-RX C→G|min-day C→G|clean四floor|
|---:|---:|---:|---:|---:|:---:|
|F1|99.2857%→99.2798%（−0.0059）|98.7857%→98.7857%（0.0000）|97.5417%→97.7500%（+0.2083）|99.1905%→99.1667%（−0.0238）|通过|
|F2|99.2619%→99.2381%（−0.0238）|98.9048%→98.8571%（−0.0476）|96.9583%→96.7917%（−0.1667）|99.0476%→99.0714%（+0.0238）|通过|
|F3|99.3631%→99.2321%（−0.1310）|98.8810%→98.5714%（−0.3095）|97.0000%→96.5000%（−0.5000）|99.2857%→99.1071%（−0.1786）|通过|
|F4|99.2679%→99.3155%（+0.0476）|98.8571%→99.0000%（+0.1429）|97.6667%→97.7083%（+0.0417）|99.1905%→99.2500%（+0.0595）|通过|
|F5|97.5655%→98.1488%（+0.5833）|93.3810%→96.0238%（+2.6429）|91.3333%→92.7500%（+1.4167）|97.3452%→97.8571%（+0.5119）|通过|
|F6|98.0417%→97.4583%（−0.5833）|95.7619%→92.8095%（−2.9524）|91.7500%→88.6250%（−3.1250）|97.8452%→96.9881%（−0.8571）|失败|

clean门仅`5/6`通过。F6的min-class和min-RX分别下降`2.9524pp`与`3.1250pp`，直接越过冻结的−2pp非补偿地板；F5的明显收益不能抵消F6失败。

六折等权clean均值为：overall从`98.797619%`到`98.778770%`（`−0.018849pp`），min-class从`97.428571%`到`97.341270%`（`−0.087302pp`），min-RX从`95.375000%`到`95.020833%`（`−0.354167pp`），min-day从`98.650794%`到`98.573413%`（`−0.077381pp`）。均值接近持平不改变逐折地板失败。

### 6.2LEO三场景与18格门

|fold|clear δoverall/四floor|low-elev δoverall/四floor|rain δoverall/四floor|三场景overall均值|通过格|
|---:|---:|---:|---:|---:|---:|
|F1|0.0000/通过|+0.9191/通过|0.0000/通过|+0.3064|3/3|
|F2|−0.9191/失败|−0.5515/通过|−1.1719/失败|−0.8808|1/3|
|F3|0.0000/通过|+0.3676/失败|+0.1950/失败|+0.1877|1/3|
|F4|−1.4706/失败|+0.3676/通过|0.0000/通过|−0.3676|2/3|
|F5|+3.4926/通过|−1.1029/失败|+4.4922/通过|+2.2940|2/3|
|F6|−3.8603/失败|−0.9191/失败|−2.1484/失败|−2.3093|0/3|

四floor单格通过`9/18`，但只有F1实现3/3，因此完整fold仅`1/6`；逐fold三场景overall非负为`3/6`。全18格等权C→G变化为：overall=`−0.128294pp`、min-class=`+0.453318pp`、min-RX=`−1.827298pp`、min-day=`−0.040998pp`，故全局overall门失败。

按场景六折等权变化为：clear的overall/min-class/min-RX/min-day=`−0.459559/−2.010995/−2.479869/−0.754481pp`；low-elev=`−0.153186/+0.245949/−2.621723/+0.309066pp`；rain=`+0.227865/+3.125000/−0.380302/+0.322421pp`。三种场景都暴露RX尾部不稳定，尤其F3、F5、F6；这不是单一场景偶然波动。

### 6.3连续proxy双门

|fold|AUROC C→G（增量）|proxy−V mean u C→G（增量）|双门|
|---:|---:|---:|:---:|
|F1|0.831939→0.839635（+0.007695）|866.017→1535.012（+668.996）|通过|
|F2|0.457251→0.383313（−0.073938）|259.189→550.974（+291.785）|失败|
|F3|0.930975→0.925916（−0.005059）|2549.691→1798.517（−751.174）|失败|
|F4|0.461584→0.478770（+0.017186）|1514.736→2009.124（+494.388）|通过|
|F5|0.869958→0.902928（+0.032971）|493.270→509.460（+16.190）|通过|
|F6|0.794137→0.795699（+0.001562）|1152.904→1205.538（+52.635）|通过|

proxy双门仅`4/6`通过。六折平均AUROC增量为`−0.003264`，平均u-gap增量为`+128.803`；F2的AUROC和F3的两项方向失败不能由其余fold补偿。

### 6.4冻结门汇总与唯一裁决

|冻结门|结果|判定|
|---|---|---|
|技术绑定|6/6 pair与F6原始工件重算通过|PASS|
|clean四floor|5/6 fold|FAIL|
|LEO四floor|9/18 cell；1/6 fold完整|FAIL|
|逐fold三场景overall|3/6非负|FAIL|
|全18格overall|`−0.128294pp`|FAIL|
|proxy双严格增益|4/6 fold|FAIL|
|Phase3 unknown能力|未评估|`NOT_EVALUATED`|

最终裁决：`REJECT_P1_CAGM_PERMANENT`。不得调`lambda_cagm`、更换fold/场景、选择F5局部收益、以平均值补偿地板或重试同一机制；CAGM不进入Phase3候选。

## 7.与ICMT的同合同复盘及下一方法约束

CAGM相对已拒绝的ICMT明显更接近目标：LEO四floor单格从`3/18`提高到`9/18`，完整fold从`0/6`提高到`1/6`，全18格overall由`−4.309002pp`收窄到`−0.128294pp`，proxy双门由`1/6`提高到`4/6`；两者clean门均为`5/6`。这表明“类内角半径＋类间质心Gram保持”比独立视图margin尾收紧更能保留整体几何，但仍没有稳定控制receiver-conditioned最坏尾部。

失败模式集中在两处：其一，F6 clean与LEO同时出现类/RX地板坍缩；其二，F3/F5/F6在LEO下的min-RX变化与overall或min-class方向分离。下一候选应直接约束source-L内可合法观测的逐接收机类内尾部，同时保持类置换对称和C/G共同forward；不能依赖target/proxy训练、跨样本事件配对、性能驱动调参，也不能再只优化全局类几何后假设RX地板自然改善。

## 8.三轮探索复盘与第四轮准入

本轮在启动第四个机制前，重新核对`项目.md`的Phase1职责与权限：`L_s`合法包含source receiver域标记`d_i`，Phase1可使用clean与LEO增强；target、V、proxy不得进入loss、optimizer、校准或选参，具体TX/RX ID不得获得专属公式或超参。项目conversation index以“GD ProtoNLL ICMT CAGM receiver RX floor postfreeze”检索未命中，因此以下只采用当前Git报告、同run pair JSON和完整日志，不用历史摘要替代。三个postfreeze目录各有19份`.out`，技术异常指纹均为0。

|探索轮|唯一新增机制|同合同真实结果|保留结论|
|---|---|---|---|
|GD-ProtoNLL|class×scene lagged风险与prototype NLL|clean5/6；LEO完整fold3/6、全18格overall`+1.500pp`；proxy双门1/6|动态风险可提高LEO平均值，但局部RX/day地板和proxy方向未被固定|
|ICMT|每类每视图低margin尾部收紧|clean5/6；LEO3/18、完整fold0/6、全18格overall`−4.309002pp`；proxy1/6|独立视图margin尾收紧在跨域弱信道下产生明显负迁移|
|CAGM|类内角半径＋类间质心Gram保持|clean5/6；LEO9/18、完整fold1/6、全18格overall`−0.128294pp`；proxy4/6|全局类几何更稳，但无法保证receiver-conditioned最坏尾部|

已永久拒绝：GD-ProtoNLL、ICMT、CAGM原机制及其同机制调参重试。剩余可证伪假设仅保留“source-L中按receiver分层、但对RX与class标签置换等价的通用尾部约束，可能把整体几何收益转成逐接收机地板收益”。第四轮必须使用同一GeoSat-C基座、同一clean＋单LEO共同forward、同一数据顺序与AdamW初态；只能读取source-L的TX与receiver标记，禁止target/proxy/V训练和按具体RX定制，并继续使用12臂训练与42步非补偿门。候选在独立P0/P1签字前不得实现或发布。
