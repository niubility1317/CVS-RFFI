# phase1_cb_sfce_postfreeze_20260809_v1实验报告

## 1. 预注册状态

- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE / NO_PERFORMANCE_RESULT`
- 日期：2026-08-09
- 负责人：`/root`；N607唯一runner：`/root/n607_geosat_lite_runner`
- 目标：对已完成的`phase1_cb_sfce12_20260809_v1`执行唯一final-only 42步闭环，按冻结非补偿门裁决P1-CB-SFCE。
- 训练输入已闭合：12/12 E40；6/6 G每折1200 batches、153600 rows、12格、未缩放梯度审计完成、terminal contract pass。
- 边界：不训练、不fit、不校准、不扫阈值、不选择checkpoint，不从proxy或LEO结果调参。

## 2. 冻结矩阵与判据

42步：12个clean export、12个source-only三场景LEO export、12个proxy连续诊断、6个CPU串行C/G pair。LEO为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，每fold同一source physical集合。

非补偿门：

1. 所有技术、checkpoint、NPZ、角色、物理ID、TX/RX、场景和顺序闭合；
2. clean 6/6 fold四个floor均`G-C>=-2pp`；
3. LEO 18/18格四个floor均`G-C>=-2pp`；
4. 18格等权overall `G-C>=0`，且6/6 fold各自三场景等权overall `G-C>=0`；
5. proxy AUROC不降且FAR不升，仅是guardrail，不得补偿clean/LEO失败。

任一门失败即`REJECT_CB_SFCE_PERMANENT`；全部通过才可列为Phase1 advancement候选，但仍不构成Phase3真实unknown能力。

## 3. 本地版本与验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`1857120bc9ebc5dc716da0ee2a1c3a58e087a221`
- 训练commit：`0f1ef07a389389156d3cb9e786a7bc278ea6ca0e`
- 独立复核：`P0=0,P1=0,ALLOW`

|文件|SHA256|
|---|---|
|`code/scripts/eval_phase1_cb_sfce_pair.py`|`80d1d0415cb862f5901bcf9a0eff825cc152dda93143bbf1fe3d1c7b5e46a841`|
|`code/scripts/launch_phase1_cb_sfce_postfreeze_20260809.sh`|`16c0574553675eac96b1c3468d09956bf37d99fec5891dd5a78097558b1163d8`|
|`code/tests/test_phase1_cb_sfce_postfreeze.py`|`03e797155287f292d73da2bb35b75f8987e87cffd41017893edc4e6d2208597a`|
|`analysis/phase1_cb_sfce_design_20260809.md`|`e194d9ae7fdf6c761acfcc50367c1e8c4b412c590d6cfde1b0f110ff2deffd48`|

验证：py_compile通过；focused pytest 20项通过；`bash -n`通过；dry-run严格为12/12/12/6共42步；`git diff --check`通过。pair evaluator使用纯NumPy和checkpoint字节SHA，不加载模型权重，避免PAMR native head加载路径。

## 4. N607冻结发布

- postfreeze run ID：`phase1_cb_sfce_postfreeze_20260809_v1`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce_postfreeze_20260809_v1_1857120b`
- training root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cb_sfce12_20260809_v1`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cb_sfce_postfreeze_20260809_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce_postfreeze_20260809_v1`
- outer log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce_postfreeze_20260809_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<release>/code`

冻结启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce_postfreeze_20260809_v1_1857120b/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_cb_sfce_postfreeze_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce_postfreeze_20260809_v1_1857120b/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cb_sfce12_20260809_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cb_sfce_postfreeze_20260809_v1_1857120b/code/scripts/launch_phase1_cb_sfce_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cb_sfce_postfreeze_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

GPU映射与训练相同：GPU0–3各2个candidate，GPU4–7各1个；pair阶段CPU串行。禁止覆盖既有training/postfreeze路径。

## 5. 健康控制与回填

- 启动后核launcher/candidate PID、CWD/cmdline、GPU、日志增长和输出计数。
- 仅协议/执行/闭合故障或至少两个distinct candidate同一确定性异常触发停止；不按性能停止。
- expected：12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON、completion/manifest/log；只回收小JSON/CSV/log，不下载checkpoint/NPZ。
- 任一export/score失败不得继续生成promotion verdict；retry=NO。

待runner回填archive/member SHA、PID/GPU、42步exit、结构计数、artifact SHA和最终非补偿门结果。
