# phase1_cb_sfce12_20260809_v1实验报告

## 1. 预注册状态

- 状态：`ARTIFACTS_COMPLETE / TRAINING_CONTRACT_COMPLETE / NO_PERFORMANCE_RESULT / POSTFREEZE_PENDING`
- 日期：2026-08-09
- 负责人：`/root`；N607唯一runner：`/root/n607_geosat_lite_runner`
- 目标：检验固定的P1-CB-SFCE是否能在不引入表示对齐、teacher、拒识head或阈值的前提下，改善有标签source-known LEO决策风险，并保持clean known稳定。
- 假设：按当前batch出现TX等权、固定`gamma=1`的satellite focal CE可缓解CCPC在困难TX和场景上的分类floor退化。
- 比较：每fold同一GeoSat-C checkpoint续训的C控制臂与G实验臂；共同`lambda_sat_cons=0.10`是冻结基座，不归因于G。

## 2. 冻结方法与边界

G唯一新增：

```text
L_G=L_C+0.10*(1/|Y_b|)*sum_c[(1/|I_c|)*sum_i((1-p^leo_i,y_i)*(-log p^leo_i,y_i))]
```

- `lambda_cb_sfce=0.10`、`gamma=1`，不调参。
- 只读既有单LEO forward的`tx_logits`和source-known local4标签。
- clear/low/rain训练场景严格round-robin；batch内出现TX等权。
- 不读取clean特征/clean logits/teacher/RX/domain/proxy/held/LEO-eval，不新增head、rejector、阈值或表示对齐。
- G首个有效batch仅做一次未缩放共享encoder/head梯度norm/cos诊断；符号不参与优化或选择。
- 终态必须闭合local4×3共12格的rows、finite loss与nonzero logit-gradient证据。

## 3. 本地版本与验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 分支：`codex/phase3-responsibility-20260807`
- 实现commit：`0f1ef07a389389156d3cb9e786a7bc278ea6ca0e`
- 独立复核：`P0=0,P1=0,ALLOW`

|文件|用途|工作树SHA256|
|---|---|---|
|`code/SSDG/train_ssdg.py`|训练接入、receipt与终态|`3890360268622cd1ac235263a1c416261c3f8709bd26eb909de705654c14d053`|
|`code/cvsrffi/phase1_cb_sfce.py`|冻结损失、绑定、审计与fail-closed|`6e4f68cdc7409670246030a0ce73f9875c37dae767e8ad8b69773eabbeeec1e7`|
|`code/tests/test_phase1_cb_sfce.py`|聚焦正负测试|`d6f9abaf6e007bc3aa46fa65f6dbc357a0370a3928b8ad1a679154b905e975bc`|
|`code/scripts/launch_phase1_cb_sfce12_20260809.sh`|12任务冻结launcher|`94d1d669ab29ddadc57cebc342c541669c7b4c3cb8b2de18cde7e95e34705291`|
|`analysis/phase1_cb_sfce_design_20260809.md`|设计追溯卡|`b19f1cde8412737e975ca8d43d58cb630c65721b78b01b956db391e53191b2e6`|

本地验证：`ssr-gpu`下py_compile通过；CB-SFCE+必要CCPC回归35项通过；真实`lite_d`无query前反向冒烟通过；训练CLI dry-run通过；launcher `bash -n`与12条dry-run通过；`git diff --check`通过。

## 4. N607冻结发布

- run ID：`phase1_cb_sfce12_20260809_v1`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce12_20260809_v1_0f1ef07a`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cb_sfce12_20260809_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce12_20260809_v1`
- outer log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce12_20260809_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基线：`runs/phase1_loto_clsgeo12_20260808_v1/F1C...F6C/final_ssdg.pth`

冻结启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce12_20260809_v1_0f1ef07a/code && nohup setsid env RUN_ID=phase1_cb_sfce12_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce12_20260809_v1_0f1ef07a/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce12_20260809_v1_0f1ef07a/code/scripts/launch_phase1_cb_sfce12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce12_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

GPU矩阵：GPU0=`F1C+F5G`；GPU1=`F1G+F5C`；GPU2=`F2C+F6G`；GPU3=`F2G+F6C`；GPU4=`F3C`；GPU5=`F3G`；GPU6=`F4C`；GPU7=`F4G`。每卡不超过2个实验。

## 5. 技术健康与产物

- 每臂40epoch、final-only、同seed、同checkpoint、同optimizer/AMP配置。
- 启动后核PID/CWD/cmdline/run-root、GPU映射、日志增长、CONFIG与首个epoch。
- 仅P0协议/安全错误、明确执行异常、无进展，或至少两个distinct arm出现同一确定性异常时停止；禁止按中间性能停止。
- 预期每臂：`metrics_epoch.csv/jsonl`、`final_ssdg.pth`、配置/终态/资源/heldout receipt；G另有CB-SFCE receipt。
- 预期P0 final gate终态可为`NON_PROMOTABLE_P0_DISABLED/exit8`；这不是技术失败。
- 正式训练完整前不启动postfreeze；本run不含性能解释。

## 6. 冻结判据

训练完成后只在固定postfreeze闭环中裁决：

1. 技术闭合；
2. clean known：6/6 fold四个floor均不低于C减2pp；
3. LEO：18个fold×scenario格的四个floor均不低于C减2pp；
4. 18格等权overall差值不低于0，且6/6 fold各自三场景等权overall差值均不低于0；
5. proxy AUROC不降且FAR不升，仅作不可补偿guardrail；严格checkpoint/artifact闭合。

任一门失败即`REJECT_CB_SFCE_PERMANENT`；不调整lambda、gamma或场景采样，不以proxy补偿held/LEO，不进入Phase3。

## 7. 运行回填

- 2026-08-09按§4冻结命令唯一启动。无prefix release archive SHA256=`d88a2f8fe8c1948633eb88f36293ca54ca5d50385671238826ecafd564182412`（261703680 bytes）；远端release成员SHA（LF归档口径）为：`train_ssdg.py=439ce211aab0a073015bb8c505ab81a1c72b0a57cf4b6214634e274059395bce`、`phase1_cb_sfce.py=ec8d4351143322647245b87790f473c2e8c6c2f5224dfc92f5c712963d333f1b`、launcher=`94d1d669ab29ddadc57cebc342c541669c7b4c3cb8b2de18cde7e95e34705291`、test=`3e110d4e6bfc2003a6badcdab824cb6fae08ff93478ee68d0dfd6ac3d322e9c4`、design=`92309a459c700e799788922af61d6bedf649c3563a574c4a46b9898abb2242d8`。归档解包后存在`release/code`且无`release/code/code`；远端py_compile、`train_ssdg.py --help`、`bash -n`和12行dry-run均通过。
- 启动caller因远端shell保持等待而超时；按AGENTS清理本地SSH后只读确认已落地，未重复启动。launcher PID=`50207`；child PID=`50210,50212,50214,50216,50218,50223,50226,50228,50233,50239,50244,50246`。物理GPU映射：GPU0=`F1C+F5G`、GPU1=`F1G+F5C`、GPU2=`F2C+F6G`、GPU3=`F2G+F6C`、GPU4=`F3C`、GPU5=`F3G`、GPU6=`F4C`、GPU7=`F4G`；结束后GPU均0%/1MiB、run进程退出。
- 12/12臂均到E040/040并写出`final_ssdg.pth`、metrics CSV/JSONL、config、terminal、training-completion、resource、heldout receipts。每臂terminal=`NON_PROMOTABLE_P0_DISABLED`、`terminal_exit_code=8`、`promotion_ready=false`，属于冻结P0 final gate预期；未发现Traceback、RuntimeError、OOM、FloatingPointError、fail-closed或参数错误。launcher脚本本身不写`completion.tsv`，该文件未产生；每臂training-completion receipt作为终态证据。
- 6个G臂的终态receipt均显示`cb_sfce_batches=1200`、`cb_sfce_rows=153600`、`cb_sfce_cells=12`、`cb_sfce_gradient_relation_attempted=true`、`cb_sfce_gradient_relation_completed=true`、`raw_unscaled=true`、`diagnostic_only=true`、`cb_sfce_terminal_contract_passed=true`，因此local4×3（每臂12格）/gradient relation合同为`6/6臂闭合`；C臂为disabled对照。这里仅报告技术合同，不解释性能。初次manifest误读了启动config receipt（其中字段为0/PENDING），现已依据`phase1_terminal_status.json`与`phase1_cb_sfce_terminal_receipt.json`更正；未重跑且未修改远端训练产物。
- 远端原始manifest SHA256=`c9357bbd5aaa68fa0b27d9f11c1d2dead6c356f1d50eee87441dd5c4c409514a`；本地更正后manifest SHA256=`4e671bd0c299c57dfce5311c4537b52c8eb27793911bf6c707fdafdd4a15e6dc`（本地路径为`artifacts\logs\phase1_cb_sfce12_20260809_v1\manifest.json`）。小artifact已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_cb_sfce12_20260809_v1\artifacts`：本地123文件（远端manifest列出的122项逐项大小/SHA全部匹配，另含manifest），不含checkpoint/NPZ；manifest记录6个C/6个G基线hash、12 child、GPU map与更正后的合同字段。性能值未读取/未解释，retry=`NO`。
- root报告与Git镜像已同步更新；Git镜像仅提交本报告修正，`git diff --check`通过。SSH/SCP/TCP22均清理，无残留连接。
