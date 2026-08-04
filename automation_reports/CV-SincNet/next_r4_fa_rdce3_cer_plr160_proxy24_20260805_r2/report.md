# NEXT-R4 FA-RDCE3×CER-PLR160 Proxy24 r2实验与分析报告

状态：`ANALYZED / SOURCE_HELD_PROXY_DIAGNOSTIC / NOT_PROMOTABLE`

## 1.实验身份与边界

- 实验ID：`next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`
- 日期：2026-08-05
- 操作者：主agent负责协议、矩阵与最终裁决；Luna/max仅负责后续已冻结发布后的机械落地与证据回收
- 候选：`NEXT-R4-FA-RDCE3-CER-PLR160`
- 协议：`p2_min_v1`
- 语义：`phase1_seen_class_loco_directional_proxy`；不构成正式Stage2-C新类注册声明
- 目标：在不改变r1科学候选、输入或矩阵的前提下，闭合r1首row的精确float32 top tie技术故障

r1已记录为：0 prediction、`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。首个逻辑行是`DA0_REG0`（域适应前/新类注册前）、qKNN基座Q、K1；异常为`TIE_UNRESOLVED: direct qKNN final float32 top tie`。r1不产生任何性能结论，r2不回写、不覆盖、不续跑r1。

## 2.四状态与冻结矩阵

|状态码|唯一中文主名称|REG0指标约束|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|`seen_new_acc`、`H_old_new`=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|`seen_new_acc`、`H_old_new`=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|报告old BA、seen-new、H、all-floor及总正确数|
|`DA1_REG1`|域适应后/新类注册后|报告old BA、seen-new、H、all-floor及总正确数|

冻结值全部沿用r1：receiver=`1-1、18-2`；每receiver 6个held-class；K=`1、5`；K1为K5逐类support前缀；逻辑行=`2×6×2=24`；唯一prediction=`144`；含K1 alias的arm记录=`192`。不改logit、参数、seed、receiver、held-class、K、阈值、矩阵、capsule、split、Phase1资产或qKNN locks。

## 3.r2唯一变化

r2只改变最终float32 top tie的确定性裁决：提交`27d46fad27f099c117dd5237e231dc8ed87871fe`实现`SEALED_CLASS_HANDLE_UTF8_ASC_V1`。每个query先在最终float32 logits上按精确`==max`形成并列集合；唯一最大值保持原预测，多个最大值仅按已封存class handle的UTF-8 bytes字典序选择最小handle。实现不改logit、不加epsilon、不读取query ID/truth/role/quota/batch class count/global assignment，也不以列索引参与排序。

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_next_r4_cer_plr160.py`|`f3c265995371653ad61aa1816bd56994f9a373b00d786cff9cb9759bffc90d37`|
|`code/cvsrffi/stage2_next_r4_runtime.py`|`130cf78da3e4d2aea5e0dfc1b32bf67924a2c6f1f94ee1057d5656e40010ffb9`|
|`tests/test_stage2_next_r4_cer_plr160.py`|`a708e23af53478495394fb8166d210d338456d348b742342f02845ef602ba9f8`|
|`tests/test_stage2_next_r4_runtime.py`|`845d48d1b1981b5b0b96c528192421119cbd2b2c19ee504cac66a0e145fc64df`|

四态比较仍固定为`DA1_REG0−DA0_REG0`、`DA1_REG1−DA0_REG1`、固定DA下的注册效应及必要的交互项。所有query仍逐样本面对全部注册类；不得读取query真值/role、真实batch类别数、class quota或global reassignment。

## 4.同一封存输入及SHA

r2沿用r1已验证输入；received-IQ字节、物理ID、receiver/TX集合、场景分配、K、support/query划分和`p2_min_v1`均不变，不触发数据重验。以下SHA全部沿用：

|输入或锁|SHA256或标识|
|---|---|
|D105 checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected received-IQ|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|received-IQ receipt|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|strict tap|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|FA asset manifest（本地）|`0dedb5ae2c6052820f44c1d9d986ff29222ac16765618cd470529671cbcb6fd8`|
|FA asset manifest（r2远端路径版）|`f858312f117a49de1b564e54189f06ceb1b213bab77c406c9627e24e626017a8`|
|capsule metadata v2|`5b30f6dda514797beb984b5ed01995cfc64b8cb5d1a7367da241a468b6ab8272`|
|validator receipt|`d0ac04930dd02a7d4c2dfe98c41d8933301e89095cc3cc5afad028f3bc499c64`|
|qKNN K1/K5 lock|`13b5496c3580b16a6660dee4fc8cd0f41874a41144b357c7c08c99b4d80e91fc`|
|Phase1 LODO authority|`b49cdc9f99094372412fd76d647cec58495a486eeb978fbd72f10e85f0f0e26a`|
|INT8 margin audit|`024a5024c06d710fbf4ddfee5326aacc89dc2ab2c74d3a9b866af09957efd9e3`|
|method-lock原始字节|`35530428ecfe77982043a3b29f3f2275c5bfb66fa1da523f64c8c01030bc7311`|
|capsule_id|`9df82b4af19898748bc5a27c039cd7e04d1f7c53fc3aa4e082c6308a3eb32a26`|
|split_id|`a5ccbba48980228a6dfb42b86116262a33184877d5ed4cafe11d406b74d05d96`|

12个FA wire、旧类聚合、qKNN锁及其逐文件SHA均按r1清单原样复用；r2不重建、不改写、不追加输入。

## 5.发布前最小证据与待落地项

|字段|r2状态|
|---|---|
|r2运行代码Git commit|`27d46fad27f099c117dd5237e231dc8ed87871fe`|
|runtime archive路径、SHA、大小|`release/next_r4_runtime_27d46fad.tar.gz`；SHA256=`54f373edad0dbf019add1172d18e5816e13ae3b81749542d1bb0a27d56f3a984`；6,529,644B；1526 members；runtime/CER/CLI成员均存在|
|远端FA manifest|`release/remote_fa_asset_manifest.json`；SHA256=`f858312f117a49de1b564e54189f06ceb1b213bab77c406c9627e24e626017a8`；12个唯一资产路径均绑定r2 root，无r1残留|
|本地聚焦验证|`ssr-gpu`下CER/runtime/artifact共21项通过；两核心文件`py_compile`通过；`git diff --check`通过|
|真实checkpoint首行smoke|R1相同predictor package、received-IQ、checkpoint和FA资产；CPU与本地`cuda:0`各完整执行首个K1 logical row的四状态，每状态Q/H均产生54个prediction，`truth_loaded=false`，两设备`tie_query_count=0`；仅证明本地路径闭合，不声称复现N607 RTX3090并列|
|独立Terra P0/P1复核|`P0=0/P1=0 / RELEASE`；独立复核在`ssr-gpu`禁用pytest cache/bytecode后16项通过，并用只读动态探针验证功能性K5最终并列、Unicode UTF-8排序和共同列+handle置换；P2仅为后续补测试/注释，不阻塞r2|
|Conda/Python、CWD、GPU|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`<fresh root>/source`；物理GPU0、`CUDA_VISIBLE_DEVICES=0`、进程内`cuda:0`，以preflight资源证据为准|
|fresh remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`；创建前必须`ABSENT`，不得复用r1 root|
|启动后PID、CWD、cmdline、GPU和日志增长证据|唯一PID=`1368667`；CWD、完整cmdline和run root绑定通过；GPU0；进程自然结束；`run.out/run.err=0B`；未重启|

上述预期artifact现已全部生成并回收；实际SHA与闭合状态见第7节。

### 5.1冻结N607执行路径

- runtime archive同步到`<fresh root>/input/next_r4_runtime_27d46fad.tar.gz`并在`<fresh root>/source`解包；12个FA wire和remote manifest同步到`<fresh root>/input/fa_assets_v1/`与`<fresh root>/input/remote_fa_asset_manifest.json`，全部逐文件校验SHA。
- capsule metadata v2同步到`<fresh root>/input/next_r4_capsule_metadata_v2.json`，SHA256=`5b30f6dda514797beb984b5ed01995cfc64b8cb5d1a7367da241a468b6ab8272`。
- received-IQ沿用只读路径`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`；checkpoint沿用只读路径`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- prepare固定调用`code/scripts/run_next_r4_proxy24.py prepare`，输入上述received-IQ/SHA、capsule/SHA、remote manifest/SHA和checkpoint SHA，输出`<fresh root>/prepare`；package、prepare receipt和truth SHA由同次prepare后机械计算并记录。
- predict固定调用`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_next_r4_proxy24.py predict --run-id next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2 --run-root <fresh root>/output --received-iq <received-IQ绝对路径> --received-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --package <fresh root>/prepare/predictor_package.json --package-sha256 <同次prepare后SHA> --fa-asset-manifest <fresh root>/input/remote_fa_asset_manifest.json --fa-asset-manifest-sha256 f858312f117a49de1b564e54189f06ceb1b213bab77c406c9627e24e626017a8 --checkpoint <checkpoint绝对路径> --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --prepare-receipt <fresh root>/prepare/prepare_receipt.json --prepare-receipt-sha256 <同次prepare后SHA> --device cuda:0`。
- 仅在prediction/completion完整后，score固定调用`code/scripts/run_next_r4_proxy24.py score --run-root <fresh root>/output --truth <fresh root>/prepare/truth.json --truth-sha256 <同次prepare后SHA> --prepare-receipt <fresh root>/prepare/prepare_receipt.json --prepare-receipt-sha256 <同次prepare后SHA> --output <fresh root>/output/score.json`。

## 6.停止规则与结果表

停止规则与r1相同：仅因协议/安全违规、错误checkout或hash、覆盖风险、prediction闭合缺失，或至少两个不同row在产生prediction前出现同一确定性异常指纹，才停止run-owned进程树并保留partial。不得按accuracy、H、BA、floor或任何性能值停止；fresh-run retry未授权。

|run|四态|prediction/score|性能指标|判定|
|---|---|---|---|---|
|`next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2`|四态完整|`144`/`score exit=0`|见第8—12节|`SOURCE_HELD_PROXY_SCORED_ONLY / NOT_PROMOTABLE`|

## 7.正式执行与artifact闭合

- N607唯一predict进程PID=`1368667`，2026-08-05 01:23:39 HKT启动，绑定物理GPU0和进程内`cuda:0`；自然结束，无第二次launch、无性能早停。
- prepare首次因runner预建空`prepare/`目录而在任何artifact前exit=1。核实精确目标位于本run且为空后，仅用非递归`rmdir`删除该空目录；同一冻结prepare命令重跑exit=0。这是机械发布修复，不是方法或性能失败。
- 完整闭合：24/24 logical rows、144/144 unique predictions、192/192 arm records；manifest/resource均24行；`rows_complete=true`、`partial_scoring_used=false`、`cross_row_best_selection_used=false`。
- predictor全程`truth_loaded=false`，query fit/update/selection均为0。truth只在prediction、manifest、resource和completion闭合后打开，`truth_label_join_only=true`。
- 无训练epoch、optimizer step或新checkpoint，因此best epoch/checkpoint=`N/A`；冻结D105 checkpoint不变。

|artifact|SHA256|
|---|---|
|predictor package|`861a92acf8ede882b7fae2ddc71c9e86098d603fbdc719ee0b24957188bc5035`|
|prepare receipt|`95c6f2cfa9fecc0c95e68275fb034e93263cd1644ca573408eacc8af403f295d`|
|truth sidecar|`48c7291271fb79da1fa9236acc5282c4aeb20b062ce591a44af733825237731b`|
|prediction|`50301712499cff744904dd0ee3791ae168e0687a770738947758b1627ba94ad0`|
|manifest|`38d85bfc12263413c1d2b7f07f6cb51d09d696b835b51c01cde67703b9dc4596`|
|resource|`9e7aa6a5517b8ded8404066bc53da763a51c36ea51fb60c31f41d4783aa4c3ed`|
|completion|`d4664e8d2466012a3d8cb71098021a717ac41a8b22f50eae172c142c751ed678`|
|score|`b96dcbaf2e004ebb557d6e321a7780c693ae255e4064f4164c6bf5e299f538c7`|

完整回收目录：`E:\type10-7\automation_reports\CV-SincNet\next_r4_fa_rdce3_cer_plr160_proxy24_20260805_r2\retrieved\`。全量日志已读：compile记录Python3.10.19、torch2.1.0+cu121、8×RTX3090；prepare、predict stderr/stdout和score日志均为空；GPU、run进程、本地SSH和N607/bridge TCP22连接均清理为0。

## 8.四状态总体性能

主表使用同一K、同一state、同一arm的12个outer row计数汇总。百分数为独立truth-side scorer输出；REG0的seen-new与H严格保持`N/A`。

|K|arm|状态|old BA|old floor|seen-new|H|all floor|forgetting|correct/query|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|1|Q|域适应前/新类注册前|74.815|54.444|N/A|N/A|N/A|N/A|404/648|
|1|Q|域适应后/新类注册前|73.333|51.111|N/A|N/A|N/A|N/A|396/648|
|1|Q|域适应前/新类注册后|72.222|50.000|72.222|72.222|50.000|2.593|468/648|
|1|Q|域适应后/新类注册后|71.296|50.000|70.370|70.830|50.000|2.037|461/648|
|1|H|四状态|与Q逐值一致|与Q逐值一致|与Q逐值一致|与Q逐值一致|与Q逐值一致|与Q逐值一致|K1 exact alias|
|5|Q|域适应前/新类注册前|80.926|61.111|N/A|N/A|N/A|N/A|437/648|
|5|Q|域适应后/新类注册前|82.593|71.111|N/A|N/A|N/A|N/A|446/648|
|5|Q|域适应前/新类注册后|79.630|55.556|79.630|79.630|55.556|1.296|516/648|
|5|Q|域适应后/新类注册后|81.481|66.667|81.481|81.481|66.667|1.111|528/648|
|5|H|域适应前/新类注册前|55.556|15.556|N/A|N/A|N/A|N/A|300/648|
|5|H|域适应后/新类注册前|55.556|16.667|N/A|N/A|N/A|N/A|300/648|
|5|H|域适应前/新类注册后|52.778|22.222|52.778|52.778|22.222|2.778|342/648|
|5|H|域适应后/新类注册后|53.333|23.333|55.556|54.422|24.074|2.222|348/648|

## 9.因果差值与组件裁决

|比较|old BA|old floor|seen-new|H|all floor|correct|裁决|
|---|---:|---:|---:|---:|---:|---:|---|
|K1 Q：域适应后−域适应前，新类注册前|−1.481pp|−3.333pp|N/A|N/A|N/A|−8|负收益；关闭K1适配|
|K1 Q：域适应后−域适应前，新类注册后|−0.926pp|0.000pp|−1.852pp|−1.392pp|0.000pp|−7|负收益；关闭K1适配|
|K5 Q：域适应后−域适应前，新类注册前|+1.667pp|+10.000pp|N/A|N/A|N/A|+9|正收益|
|K5 Q：域适应后−域适应前，新类注册后|+1.852pp|+11.111pp|+1.852pp|+1.852pp|+11.111pp|+12|联合正收益；保留FA-RDCE3 K5|
|K5 H−Q：域适应前/新类注册后|−26.852pp|−33.333pp|−26.852pp|−26.852pp|−33.333pp|−174|CER头显著负收益|
|K5 H−Q：域适应后/新类注册后|−28.148pp|−43.333pp|−25.926pp|−27.060pp|−42.593pp|−180|CER头显著负收益；立即淘汰|
|`(DA1_H−DA1_Q)−(DA0_H−DA0_Q)`，注册后|−1.296pp|−10.000pp|+0.926pp|−0.208pp|−9.259pp|−6|DA不能挽救CER头|

稳定性上，K5 Q的新类注册前old BA在12行中6正、6平、0负；新类注册后H为9正、3平、0负。K1 Q的新类注册前old BA为1正、6平、5负；新类注册后H为0正、7平、5负。K5 CER头的新类注册后H为12/12负。因此这里不是“继续调CER参数”的信号，而是明确的结构性拒绝证据。

## 10.receiver与24行同row结果

### 10.1 receiver级Q结果

|K|receiver|状态|old BA|old floor|seen-new|H|all floor|forgetting|correct/query|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|1|1-1|域适应前/新类注册前|87.037|68.889|N/A|N/A|N/A|N/A|235/324|
|1|1-1|域适应后/新类注册前|87.407|68.889|N/A|N/A|N/A|N/A|236/324|
|1|1-1|域适应前/新类注册后|85.185|66.667|85.185|85.185|66.667|1.852|276/324|
|1|1-1|域适应后/新类注册后|85.185|66.667|85.185|85.185|66.667|2.222|276/324|
|1|18-2|域适应前/新类注册前|62.593|8.889|N/A|N/A|N/A|N/A|169/324|
|1|18-2|域适应后/新类注册前|59.259|2.222|N/A|N/A|N/A|N/A|160/324|
|1|18-2|域适应前/新类注册后|59.259|0.000|59.259|59.259|0.000|3.333|192/324|
|1|18-2|域适应后/新类注册后|57.407|0.000|55.556|56.466|0.000|1.852|185/324|
|5|1-1|域适应前/新类注册前|85.185|68.889|N/A|N/A|N/A|N/A|230/324|
|5|1-1|域适应后/新类注册前|87.778|68.889|N/A|N/A|N/A|N/A|237/324|
|5|1-1|域适应前/新类注册后|83.333|66.667|83.333|83.333|66.667|1.852|270/324|
|5|1-1|域适应后/新类注册后|85.926|66.667|85.185|85.554|66.667|1.852|278/324|
|5|18-2|域适应前/新类注册前|76.667|48.889|N/A|N/A|N/A|N/A|207/324|
|5|18-2|域适应后/新类注册前|77.407|53.333|N/A|N/A|N/A|N/A|209/324|
|5|18-2|域适应前/新类注册后|75.926|44.444|75.926|75.926|44.444|0.741|246/324|
|5|18-2|域适应后/新类注册后|77.037|51.111|77.778|77.406|51.852|0.370|250/324|

### 10.2 held-class同rowQ结果

斜线四元组依次为`old BA/seen-new/H/all floor`；差值三元组依次为域适应后的`Δold/Δnew/ΔH`。

|receiver|held class|K|域适应前/注册前old|域适应后/注册前old|注册前DA差|域适应前/注册后四元组|域适应后/注册后四元组|注册后DA差|K5 H的注册后H|H−Q的H差|
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
|1-1|14-10|1|82.22|82.22|+0.00|82.22/100.00/90.24/66.67|82.22/100.00/90.24/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|1-1|14-7|1|93.33|93.33|+0.00|88.89/66.67/76.19/66.67|88.89/66.67/76.19/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|1-1|20-15|1|82.22|82.22|+0.00|82.22/100.00/90.24/66.67|82.22/100.00/90.24/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|1-1|20-19|1|91.11|93.33|+2.22|86.67/77.78/81.98/66.67|86.67/77.78/81.98/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|1-1|6-15|1|84.44|84.44|+0.00|82.22/100.00/90.24/66.67|82.22/100.00/90.24/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|1-1|8-20|1|88.89|88.89|+0.00|88.89/66.67/76.19/66.67|88.89/66.67/76.19/66.67|+0.00/+0.00/+0.00|N/A|N/A|
|18-2|14-10|1|71.11|71.11|+0.00|71.11/0.00/0.00/0.00|71.11/0.00/0.00/0.00|+0.00/+0.00/+0.00|N/A|N/A|
|18-2|14-7|1|60.00|57.78|−2.22|55.56/77.78/64.81/0.00|53.33/77.78/63.28/0.00|−2.22/+0.00/−1.54|N/A|N/A|
|18-2|20-15|1|62.22|53.33|−8.89|55.56/77.78/64.81/0.00|53.33/77.78/63.28/0.00|−2.22/+0.00/−1.54|N/A|N/A|
|18-2|20-19|1|71.11|68.89|−2.22|62.22/44.44/51.85/0.00|62.22/22.22/32.75/0.00|+0.00/−22.22/−19.10|N/A|N/A|
|18-2|6-15|1|55.56|51.11|−4.44|55.56/77.78/64.81/0.00|51.11/77.78/61.69/0.00|−4.44/+0.00/−3.13|N/A|N/A|
|18-2|8-20|1|55.56|53.33|−2.22|55.56/77.78/64.81/0.00|53.33/77.78/63.28/0.00|−2.22/+0.00/−1.54|N/A|N/A|
|1-1|14-10|5|80.00|84.44|+4.44|80.00/100.00/88.89/66.67|84.44/100.00/91.57/66.67|+4.44/+0.00/+2.68|80.37|−11.20|
|1-1|14-7|5|93.33|93.33|+0.00|86.67/66.67/75.36/66.67|86.67/66.67/75.36/66.67|+0.00/+0.00/+0.00|72.73|−2.64|
|1-1|20-15|5|80.00|82.22|+2.22|80.00/100.00/88.89/66.67|82.22/100.00/90.24/66.67|+2.22/+0.00/+1.36|83.12|−7.13|
|1-1|20-19|5|88.89|88.89|+0.00|86.67/66.67/75.36/66.67|86.67/77.78/81.98/66.67|+0.00/+11.11/+6.62|66.31|−15.67|
|1-1|6-15|5|82.22|86.67|+4.44|80.00/100.00/88.89/66.67|84.44/100.00/91.57/66.67|+4.44/+0.00/+2.68|84.62|−6.95|
|1-1|8-20|5|86.67|91.11|+4.44|86.67/66.67/75.36/66.67|91.11/66.67/77.00/66.67|+4.44/+0.00/+1.63|71.79|−5.20|
|18-2|14-10|5|73.33|73.33|+0.00|73.33/88.89/80.37/44.44|73.33/88.89/80.37/44.44|+0.00/+0.00/+0.00|16.93|−63.43|
|18-2|14-7|5|77.78|77.78|+0.00|75.56/77.78/76.65/44.44|75.56/77.78/76.65/44.44|+0.00/+0.00/+0.00|0.00|−76.65|
|18-2|20-15|5|75.56|75.56|+0.00|73.33/88.89/80.37/44.44|75.56/88.89/81.68/55.56|+2.22/+0.00/+1.32|26.67|−55.02|
|18-2|20-19|5|82.22|82.22|+0.00|82.22/44.44/57.70/44.44|82.22/55.56/66.31/55.56|+0.00/+11.11/+8.61|0.00|−66.31|
|18-2|6-15|5|75.56|77.78|+2.22|75.56/77.78/76.65/44.44|77.78/77.78/77.78/55.56|+2.22/+0.00/+1.13|31.82|−45.96|
|18-2|8-20|5|75.56|77.78|+2.22|75.56/77.78/76.65/44.44|77.78/77.78/77.78/55.56|+2.22/+0.00/+1.13|31.82|−45.96|

## 11.资源、并列裁决与数据QA

|组件|K1|K5|
|---|---:|---:|
|FA动态数值态|6B|6B|
|FA support fit MAC/row|2,400|12,000|
|FA每query固定MAC|960|960|
|qKNN数值态，5/6类|830/996B|4,110/4,932B|
|qKNN每query MAC，5/6类|800/960|4,000/4,800|
|CER新增态，5/6类|0/0B|820/984B|
|CER新增每query MAC，5/6类|0/0|800/960|
|CER解析fit MAC，5/6类|0/0|18,880/22,400|
|epoch/optimizer/query-fit|0/0/0|0/0/0|

DA1_REG1对DA1_REG0的FA状态在24/24行均为同一对象且逐bit SHA一致；新类support用于DA的行数为0，确保注册效应不反向污染域适应状态。

N607实际出现4个唯一K1`DA0_REG0`query并列，分布在3个outer row；Q/H因exact alias各自记录相同并列，不能重复计成8个query。全部由`SEALED_CLASS_HANDLE_UTF8_ASC_V1`闭合，其他state/K均0并列。这证明r1技术故障被真实N607路径修复，而非只在合成测试中通过。

主agent对完整score/prediction/truth执行12项独立复算，全部通过：score SHA、24行闭合、192个state-arm唯一键、108个唯一truth query、truth后开、总体聚合、因果差值、K1 alias、query隔离、tie receipt和FA状态复用。runner生成的`evidence/score_structure.txt`写`truth_query_count_opened=588`，但权威`score.json`字段、`truth.json`键数和prediction中query ID并集均严格为108；588是received capsule总物理记录数，不是打开的唯一query数。该证据摘要笔误不影响预测或评分，但报告统一纠正为108。

## 12.与D62、D91、D92及D92-Lite的边界化比较

下表只作机制与证据层级对照。历史Target125使用5receiver×5seed×5slice；NEXT-R4使用2receiver×6 held-class×K1/K5的source-held Proxy24，不能按绝对百分数直接排名。

|方法|矩阵与证据|域适应前/注册前old|域适应前/注册后old|注册后seen-new|注册后H|注册后floor|结论|
|---|---|---:|---:|---:|---:|---:|---|
|D62|历史完整Target125诊断|81.51|64.39|59.11|61.09|35.15|遗忘17.11pp；关闭|
|D91|7候选×15 outer development|N/A|N/A|N/A|N/A|N/A|无独立raw score；matched prediction回退D62；不晋级|
|D92|历史完整Target125诊断|未在当前摘要统一回收|65.56|58.93|61.57|未在当前摘要统一回收|相对D62小幅改善旧类/H但新类−0.18pp；仍不晋级|
|SVRN-qKNN-BCRR/r4.2|历史完整Target125诊断|73.10|43.03|23.46|29.25|11.21|相对D62的H−31.84pp；永久关闭|
|D92-Lite-FULL288/r1|2026-08-04完整Target125诊断-only|78.59|60.35|41.96|51.21|32.49|遗忘18.24pp；仅M_JOINT，无四态DA覆盖|
|NEXT-R4 K1 Q|本run source-held Proxy24|74.82|72.22|72.22|72.22|50.00|DA后H降至70.83；关闭K1适配|
|NEXT-R4 K5 Q|本run source-held Proxy24|80.93|79.63|79.63|79.63|55.56|DA后四项均为81.48、floor66.67；保留为机制正信号|
|NEXT-R4 K5 H|本run source-held Proxy24|55.56|52.78|52.78|52.78|22.22|DA后H仅54.42；CER头淘汰|

D92与NEXT-R4回答不同问题：D92在注册后重配旧/新协方差，曾以新类轻微下降换取旧类与遗忘改善；NEXT-R4的FA-RDCE3 K5在本Proxy24中同时提高注册前old、注册后old、seen-new、H和floor，没有观察到旧/新交换。但这只是两receiver source-held正信号，不能覆盖D92的Target125证据，更不能声明正式target性能已提升。

## 13.最终裁决与下一方法

1.`CER-PLR160`在K5的12/12 held-row上降低H，聚合损失27.060pp，并增加820—984B状态与800—960 MAC/query。该组件确定淘汰，不重跑、不调rank、不调残差权重。
2.`FA-RDCE3 K1`整体负收益，尤其receiver18-2注册后floor为0；K1缺乏稳定的域尺度估计，不继续开发。
3.`FA-RDCE3 K5＋direct qKNN`是本轮唯一正收益版本：域适应后/新类注册后old、seen-new和H各+1.852pp，all floor+11.111pp，12行中H为9正3平0负。保留它作为下一候选的DA核心，但维持`SOURCE_HELD_PROXY_SCORED_ONLY`，不晋级。
4.下一方法不扩大矩阵。科学路线冻结为：保留K5 FA-RDCE3和直接qKNN；分类头改为“默认严格等于Q、仅在support-held可证明不降低old/new下界时才产生类对称小残差”的保守头，首先用Phase1/source-held反证其是否会重现CER的receiver18-2崩塌。若不能在不增加query依赖、阈值扫描或角色信息的条件下证明作用，直接保持无头Q，不再为“联合D92”强行增加计算。
