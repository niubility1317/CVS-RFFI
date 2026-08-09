# Phase1 GD-ProtoNLL v3一次性postfreeze报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标与冻结边界

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll_postfreeze_20260810_v1`|
|目标|对已完成的GD-ProtoNLL v3 6折C/G final checkpoint执行一次性42步postfreeze，机械产生唯一接受或永久淘汰结论|
|训练输入|`phase1_gd_proto_nll12_20260809_v3`的12个`F{fold}{C/G}_GD_PROTO_NLL12/final_ssdg.pth`，训练合同已完整通过|
|实现commit|`a3a97d882a4bede06e638c9cb1a5697b8e3f459d`|
|独立复核|`P0=0、P1=0、ALLOW`；本地pycompile、focused 9、bash-n、dry-run42、diff-check全部通过|
|禁止|不训练、不校准阈值、不选择checkpoint/折/场景；失败不调参、不重试、不以proxy补偿分类门|

冻结文件SHA256：pair evaluator=`9daa7df9d94cdafdaa648ce9ba5fb03a88847a4686fcfe6daa9a9e895b3d3b76`；GD clean exporter=`c21d84a76c5448e0c8414389222ad764575820f011b05784982d313499f71580`；launcher=`c6ea80a3112c235761c94ab26be03cd2dfd9e872ba0390684300313838490e30`；test=`0cd3d64da0f9a70c90f8ea6b01f5bb19bab25eec77170af0958b2cc5b6652b30`；design=`91b8800bed75f012dd23d21ad37c5e8c643b4bc774b1f72241c7c12d3f5eccff`。

## 2.42步数据与判定

每个候选一次GD-clean导出严格重建训练同一local4、seed=`7281105`、split=`tx_rx_day_1_6_3`、L/U/V=`.07/.63/.30`：只forward L、V与proxy，U forward/persist均为0；ManySig实际SHA必须为`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。clean分类只用checkpoint未见V；Gaussian只用L拟合，known u只用V，unknown u只用proxy；无阈值。

LEO继续使用通用source-only导出：local4×400=1600行，三场景`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，runtime view=`single`、source profile=`satellite`、TTA=`none`、seed=`7281718`。clean V和LEO是两个明确不同的固定诊断slice，各自C/G物理行严格配对，不混写。

判定为非补偿硬门：clean 6/6四floor均不低于C−2pp；LEO 18格四floor均不低于C−2pp；每折三场景等权overall Δ≥0；18格等权overall Δ≥0；逐折6/6同时满足GD连续u的ΔAUROC>0与Δ(proxy mean u−V-known mean u)>0。任一失败即`REJECT_GD_PROTO_NLL_PERMANENT`。

## 3.N607路径、矩阵与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88`|
|CWD|`<release>/code`|
|training root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`，只读|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll_postfreeze_20260810_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1_launcher.out`|
|矩阵|0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`|
|步骤|12 GD-clean＋12 LEO＋12 proxy＋6 pair=42|
|retry|NO|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code && nohup env POSTFREEZE_RUN_ID=phase1_gd_proto_nll_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v1_a3a97d88/code/scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &
```

## 4.发布与健康门

唯一runner先direct preflight；启动前核release/run/log/outer均ABSENT、训练root12 checkpoint存在、ManySig与archive/member SHA、远端pycompile/help/bash-n/dry-run42、GPU每卡≤2。只启动一次；caller超时后只读核验，不重发。

停止只依据路径/hash/覆盖、split/head/checkpoint/arm/root绑定、Traceback/OOM/CUDA、确定性异常、缺失输出或零pair；不得依据任何性能数值提前停。结束后需有12 GD-clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON及日志；只回收小JSON/CSV/log/manifest/completion，不下载NPZ/checkpoint。最终由F6 aggregate原样给出结论，runner只作技术核验。

## 5.落地、启动前静态核验（2026-08-10 00:41 CST）

- direct `N607` preflight通过：项目根、服务器时间与GPU可见；GPU7约498MiB（SCB v3 PID=`420208`约488MiB），GPU0–6各约1MiB；本地每次SSH/SCP后`ssh.exe=0`且N607/bridge TCP22为0。
- 新release、postfreeze run/log/outer及远端临时包在落地前均为`ABSENT`；只读training root保持12个`final_ssdg.pth`，未修改或下载checkpoint。
- ManySig远端SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C baseline full SHA与冻结值一致：F1C=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`、F2C=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`、F3C=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`、F4C=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`、F5C=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`、F6C=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。
- 由实现commit=`a3a97d882a4bede06e638c9cb1a5697b8e3f459d`生成无prefix archive：本地`263137280` bytes、SHA256=`4bc1cc6d448b32af93f2c0a8878cb9f0f044ce81e466c9aebeddc059c7be7df5`、`4884` members、`code/code=0`；远端临时包同SHA/大小/成员数并解包至新release。远端LF member SHA：pair=`3581cc2c488957c7cf2010e92bab4edfa266ce2f004df84cefa2bd1b2b1650bc`、GD-clean exporter=`6f75e19352ec400a5f64ee900ec476f09894974ccdf12b243eaf60a18b175874`、launcher=`c6ea80a3112c235761c94ab26be03cd2dfd9e872ba0390684300313838490e30`、test=`b089f8316657fa2edb6208e9d5bb29f6a3e0d1b7651b32c08ff0175a3aea1530`、design=`c91b8721fc75d4ea71a314246836696154a32ce64ca2ba22669ad95d515c549a`。
- 远端静态核验通过：相关Python模块/测试`py_compile`、GD exporter/pair/proxy `--help`、launcher `bash -n`；`--dry-run`精确42行，GD-clean=12、LEO=12、proxy=12、pair=6，dry-run bytes=`55470`、SHA256=`79393c26b820ee21cf8e571225900508224f74fe2b333458ac8a3c30f4335896`。静态临时文件已删除并核验`ABSENT`。

启动前未执行唯一命令；随后严格按本报告§3显式`bash`命令启动一次，并完成launcher/12 children/CWD/cmdline/GPU/log绑定核验。

## 6.唯一启动、技术停止与partial回收（2026-08-10）

- 严格按§3命令显式调用`bash`启动一次，SSH返回`0`，未重发；矩阵PID表记录12个primary child：GPU0=`523859(F1C),523860(F5G)`、GPU1=`523861(F1G),523862(F5C)`、GPU2=`523864(F2C),523865(F6G)`、GPU3=`523867(F2G),523868(F6C)`、GPU4=`523870(F3C)`、GPU5=`523873(F3G)`、GPU6=`523875(F4C)`、GPU7=`523879(F4G)`。启动后12个child及launcher/runner均已退出，未执行kill；GPU0–6回到空闲基线，GPU7仅保留SCB PID=`420208`约488MiB。
- 首波技术核验确认`F6_C_vs_G_pair.out`出现冻结停止指纹：`Traceback`最终为`GDProtoNLLPostfreezePairError: geometry score features contain non-finite or zero L2 norm`。这是技术失败门，保留partial并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不重试、不启动postfreeze v2、不读取或解释性能字段。outer launcher为`0` bytes。
- 远端partial计数：GD-clean NPZ=`12`（留在N607、未下载）、LEO NPZ=`12`（留在N607、未下载）、proxy JSON/CSV=`12/12`、pair JSON=`5/6`（F6缺失）。仅回收49-member小bundle（无`.npz/.pth`）及技术诊断JSON；小bundle=`3848959` bytes、SHA256=`4fe0b5c785ff8099a19b13747e6426de93a414540bcaea2273080e97eee047e2`；诊断JSON=`10330` bytes、SHA256=`5fe83c95b2b23c7c3271ffab31451fcacf957087dd1d7a82c33f4b697d7d983c`。逐项清单`remote_artifact_sha256.tsv`共52 rows、`8648` bytes、SHA256=`92cb4dbe93f1ec25a806f4140ba123098d6cd53ebcc38d78c4079d5d70bcf1bd`；`completion.tsv`=`1730` bytes、SHA256=`2a6f2baba3bbb6326f125ece34a5fb1ad6a4e1156c8cbe633180a3a05cec364f`；`manifest.json`=`7652` bytes、SHA256=`db93751563fbe34e61e8783488918e375c9ab4b1e0212f5c9e9d4633ca2f5af4`；`runner_handoff.json`=`2576` bytes、SHA256=`b63221cccc11ee89ca6674b66d8843a7b145224c0f4ebc27cf75d3f5471bba94`。
- NPZ只读技术诊断（未下载NPZ、未输出特征值）：12个GD-clean均`21120×160`且nonfinite行=`0`；仅`F6C_GD_PROTO_NLL12`含1个zero-norm行，归属`source_validation_known`（该role=`16800`行、zero=`1`；`labeled_fit`=`3920/0`、`proxy_unknown`=`400/0`），F6G及其余11候选三role zero均为0。
- 远端临时archive/list/static/bundle/diagnostic均已删除并核验`ABSENT`；run-owned进程=`0`；本地每次SSH/SCP后`ssh.exe=0`、N607/bridge TCP22=`0`。本run无性能结论；后续由主控按技术失败证据处理，不由runner重启或发起新run。
