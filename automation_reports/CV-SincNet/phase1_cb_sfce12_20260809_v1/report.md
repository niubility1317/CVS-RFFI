# phase1_cb_sfce12_20260809_v1实验报告

## 1. 预注册状态

- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE / NO_PERFORMANCE_RESULT`
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

待runner回填release archive SHA、远端成员SHA、launcher/child PID、实际GPU占用、completion、产物哈希、异常与清理状态。当前无N607性能结果。
