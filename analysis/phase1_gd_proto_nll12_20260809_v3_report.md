# Phase1 GD-ProtoNLL 12臂正式训练v3报告

状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标与唯一方法变化

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll12_20260809_v3`|
|目标|完成P1-GD-ProtoNLL 6折C/G、40E、final-only 12臂矩阵，生成一次性postfreeze所需的12个最终checkpoint|
|v2终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；F1G在E009由合并错误`feature/head contains non-finite or zero L2 norm`停止，通用fingerprint不能判定具体分支|
|v3唯一科学修复|GD辅助项只过滤精确有限零范数feature行；feature非有限、head非有限或零范数仍fail-closed；过滤后每个local4类至少1个valid行|
|不变部分|C路径、完整batch的base与共同`sat_cons=.10`、四类等权、`3Σqℓ`、旧q反传后EMA与12格softmax、数据、seed、40E、矩阵、final-only|
|实现commit|`10ead8b299f26ac793941180bee1bc87c45aa4df`|
|独立复核|科学摘要与实际diff均为`P0=0、P1=0、ALLOW`|
|声明|本run不以训练中性能停止，不在训练阶段读取或解释性能；完整postfreeze前不作晋级结论|

本地冻结SHA：core=`F2AAB82148F882B2BF7C6DEC0886396157720BFB720A41DF70552A84926632E4`；train=`FD571751B70EC74ACF38A04E5288092C085A3ECE97F8E99FD79EF7D3C7E1F67F`；test=`3E1A0EE72DC10800493C7AA782521F5A2C562299BFB3D45DC17E7CE35ECB026A`；launcher=`BE109FF1F209CC928E0C437354860A95C99D1926D609DECF2E4B5609C50E05A1`；design=`6CB62DBF9AC3C10236FBAF655A239ACCE5E96989CC0911335B96420176BF5649`。`py_compile`、GD+CB+CP 31项、3个lite_d no-query smoke、`bash -n`、dry-run12和`git diff --check`均通过。

## 2.数据、矩阵与合同

ManySig=`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`、seed=`7281105`、L/U/V=`.07/.63/.30`；GD只读L，U关闭，V仅final-only评分。6个GeoSat-C基线路径位于`runs/phase1_loto_clsgeo12_20260808_v1/F{1..6}C_LOTO_CLSGeo12/final_ssdg.pth`，SHA依次为`4d515204...c040`、`29c7d7ca...76d`、`39c6cdd6...4b7`、`32d956f4...5e4`、`2b938154...a9f`、`573ca9d0...de6f`；runner须在启动前核全长SHA。

GPU矩阵固定：0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。GPU7已有SCB v3构建时允许加入F4G，但任一卡不得超过2个计算进程。

G的receipt逐batch及12格记录`total_rows=valid_rows+zero_rows`、valid loss、有限性和解析非零logit-gradient见证；过滤前后均需local4。zero计数只作诊断，不参与阈值、选择或调参。C为`CONTROL_ARM_NOT_APPLICABLE`。

## 3.N607不可覆盖路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；caller超时后只读核验landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code && nohup env RUN_ID=phase1_gd_proto_nll12_20260809_v3 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v3_10ead8b2/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v3_launcher.out 2>&1 < /dev/null &
```

## 4.发布、健康停止与预期工件

唯一runner先执行direct preflight；失败时只用verified bridge。启动前核release/run/log/outer均ABSENT、commit/archive/member、ManySig及6个基线SHA、SCB进程和每卡≤2；远端运行`py_compile`、help、`bash -n`和dry-run12后只启动一次。

启动后记录launcher/12 child的PID、CWD、cmdline、GPU与日志增长。只按路径/hash/覆盖错误、协议或类序漂移、Traceback/OOM/CUDA、拆分后的`FEATURE_NONFINITE`、`HEAD_NONFINITE_OR_ZERO`、`FEATURE_ZERO_FILTERED_INVALID_COVERAGE`、两行同一确定性异常、零checkpoint或成员缺失停止；不得按accuracy、loss趋势或任何性能值停止。触发时只终止严格绑定本run的进程树，保留partial，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

成功预期12臂E40、final checkpoint、metrics/config、GD terminal/training/terminal/resource/heldout receipts。G应为1200 attempted batches，`state_update_batches=batches`，全部batch过滤后local4有效，12格计数闭合且终态通过；最终`NON_PROMOTABLE_P0_DISABLED/exit8`为预期训练P0门。只回收小日志、JSON/CSV、PID、completion和manifest，不下载checkpoint/NPZ；完整postfreeze另行执行。

## 5.Runner预检、落地与静态验证（启动前）

2026-08-09直连`N607`预检通过（23:43:07 CST）；GPU0–6各1MiB，GPU7约498MiB。v3 release、run、log、outer启动前均为`ABSENT`；v1/v2 release及outer仅只读保留。GPU7现有SCB v3 compute PID=`420208`、约488MiB；未发现GD/ProtoNLL进程。ManySig远端SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C baseline全长SHA与§2及v1一致：F1C=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`、F2C=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`、F3C=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`、F4C=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`、F5C=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`、F6C=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。

由实现commit=`10ead8b299f26ac793941180bee1bc87c45aa4df`生成无prefix archive：262983680 bytes、SHA256=`8a1738ef6e8902a92310b60666dadd335e1e1adcea16a1e3875a81cf75d5b39e`、4879 members、`code/code=0`。远端临时包同SHA/大小并解包到新release；远端5个LF member SHA为`phase1_gd_proto_nll.py=a06e6d958f164bde3cc5f30822524d550f9b105c149fe546dacdc2d012c72d14`、`train_ssdg.py=a47cc87337406225f1892dd0a741da4ffea6e12709ee15847016d596e7987ee9`、`test_phase1_gd_proto_nll.py=b089f8316657fa2edb6208e9d5bb29f6a3e0d1b7651b32c08ff0175a3aea1530`、`launch_phase1_gd_proto_nll12_20260809.sh=be109ff1f209cc928e0c437354860a95c99d1926d609decf2e4b5609c50e05a1`、`phase1_gd_proto_nll_design_20260809.md=9780b5b03b324f3ff6bf1030c21892f17df290a07f8814569d4ce45af6c5d5dc`。launcher解包mode=`664`，正式命令显式调用`bash`；远端`py_compile`、`train_ssdg.py --help`（92102 bytes，GD参数命中2）、`bash -n`、精确12行dry-run均PASS，dry-run SHA=`df52c77bbaef1ae5fc6f509c48fe6a9460c42e10c998a467fe7339b8359951f9`。临时archive已删除并核验`ABSENT`；每次SSH后本地`ssh.exe=0`、N607/bridge TCP22均为0。

## 6.唯一启动、技术闭合与小工件（无性能结论）

按§3 exact命令仅启动一次（23:47:53）；调用方约34秒超时后仅只读确认落地，识别并关闭了对应本地SSH PID=`30524`，未重发。wrapper PID=`486669`、launcher PID=`486670`、run PGID=`486668`；`pids.tsv`记录12个primary child：GPU0=`F1C:486673、F5G:486675`，GPU1=`F1G:486677、F5C:486679`，GPU2=`F2C:486681、F6G:486683`，GPU3=`F2G:486686、F6C:486688`，GPU4=`F3C:486690`，GPU5=`F3G:486692`，GPU6=`F4C:486694`，GPU7=`F4G:486696`。每个CWD均绑定v3 release/code，cmdline含v3 run ID和对应candidate；启动及运行期间每卡计算进程不超过2，GPU7为F4G加SCB v3 PID=`420208`。

12个臂均自然闭合，launcher/runner进程最终为0；远端计数final checkpoint=`12`、terminal类receipt=`24`、failure receipt=`0`。每臂`metrics_epoch.csv`为41行、`metrics_epoch.jsonl`为40行，12个config、heldout、resource、training completion和terminal status receipt均存在。C臂按冻结合同为`CONTROL_ARM_NOT_APPLICABLE`；6个G臂技术合同均为：`gd_batches=1200`、`total_rows=153600`、12 cells、`state_update_batches=1200`、`all_local4_valid_batches=1200`、analytic raw-logit-gradient witness attempted/completed、terminal contract pass；F1G为`valid_rows=153599、zero_rows=1`，其余G为`valid_rows=153600、zero_rows=0`。12个terminal receipt均为预期`NON_PROMOTABLE_P0_DISABLED/exit8`，`performance_result_available=false`；该P0终态是冻结训练门，不是性能结果，未读取或解释loss/accuracy/任何性能字段。

仅回收允许的小工件：远端bundle 121 members、17090560 bytes、SHA256=`defe2b27d4dfdd1d71395b83356a01020a26b18798b7f8f95163564a0f2f2844`；本地解包121 files、16991540 bytes、禁`.pth/.npz`计数=`0`。逐项SHA清单`remote_artifact_sha256.tsv`共123 rows（含archive与outer）、19250 bytes、SHA256=`567b5ce1acb20d60180c96f7f1b50d618e370fe5d6b68553783d56b4ebae4e92`；outer launcher为0 bytes、SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。completion.tsv=`378 bytes、SHA256=4b13b63680558c8ca60ace098047bf15ebcd498f9bb2c0e6705777fc66cf98ad`；runner_handoff.json=`2872 bytes、SHA256=d65e4aed169ea8d2dc8cc297e5aaa0ed73501876b3b101bfa2c816fada7e47db`；manifest.json=`4212 bytes、SHA256=f2ace0398eb89486afd7c8475709c61e6bfd12ba945f43ed0ff346a01cd66126`。远端archive、small bundle和file list临时路径均已删除并核验`ABSENT`；最终GPU0–6释放为约1MiB，GPU7仅SCB约488MiB，SCB未干预；本地`ssh.exe=0`、N607/bridge TCP22均为0。完整checkpoint不下载，postfreeze性能分析另行执行。
