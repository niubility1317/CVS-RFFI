# Phase1 CAGM后冻结42步v2预登记报告

状态：`LOCAL_VERIFIED / TRAINING_ARTIFACTS_COMPLETE / PREREGISTERED_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

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

