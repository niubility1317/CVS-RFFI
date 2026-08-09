# phase1_cb_sfce_postfreeze_20260809_v1实验报告

## 1. 预注册状态

- 状态：`ANALYZED / REJECT_CB_SFCE_PERMANENT / NO_PHASE1_ADVANCEMENT / NO_PHASE3_PROMOTION`
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

- 2026-08-09按冻结命令唯一启动，launcher PID=`79633`；candidate PID/GPU记录为`79646:F1C@0,79648:F5G@0,79649:F1G@1,79651:F5C@1,79652:F2C@2,79654:F6G@2,79656:F2G@3,79658:F6C@3,79660:F3C@4,79664:F3G@5,79667:F4C@6,79669:F4G@7`。launcher与所有child均已退出，GPU回到0%/1MiB；无SSH/TCP22残留。
- 无prefix release archive SHA256=`6911dfa4b616574b526cc44776e89a8cfb2f07918d31f93dc0b24bd35a77cc20`（261795840 bytes）。远端成员SHA（LF归档口径）：`eval_phase1_cb_sfce_pair.py=bd3fec8c41001933fd69f03b70b0a17843afa30cc3a71f221982ff802c6147c2`、launcher=`16c0574553675eac96b1c3468d09956bf37d99fec5891dd5a78097558b1163d8`、`export_spaceborne_features.py=5bd7b4fa184741b9918453eb8bf0773e6a84032da634954ee339f3576124f6fe`、logits scorer=`ac1cf8e45fadbc0782282625300a0c30936fdce4b2df241558dff979dafe5c04`、postfreeze test=`8a0d86fee3fdb1c29517767293cdf222ea3e73608e0639ae63524bc908c4861a`、design=`cb33ebf5a0f6c13b952daddd63b639dedd8ec52277eb4a2b92a702e82bf42b4c`；py_compile/help/bash-n通过，dry-run=`42`。
- 42步技术产物闭合：clean export=`12/12`、source-only LEO export=`12/12`、proxy score=`12/12`、CPU pair=`6/6`，均按完成闭环推断exit 0；日志无Traceback/RuntimeError/OOM/CUDA/FileNotFound/ValueError指纹。clean每候选2400行（source1600/target_old400/proxy_unknown400），LEO每候选1600行（source-only、`channel_views=single`、三场景、两天、六RX），各自物理row-binding unique且checkpoint SHA strict binding为12/12；pair technical binding为6/6。F6 aggregate机械字段：fold_indices=`[1,2,3,4,5,6]`、prior_pair_count=`5`、technical_binding=`true`、`phase3_unknown_capability_claim=NOT_EVALUATED`；pair JSON中的aggregate verdict字段原样为`REJECT_CB_SFCE_PERMANENT`，runner不解释该字段。
- 小artifact回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_cb_sfce_postfreeze_20260809_v1\artifacts`：远端小tar SHA256=`aeee6e0433890e239b3913cd6204693f87403062c4b2c0be804fb7873a427fcd`（528714 bytes，49项）；本地共51项（49远端文件+本地completion/manifest），不含NPZ/checkpoint。`manifest.json` SHA256=`2dfd40cc855893d82265796c38387fa31a1edfdc0301a2416e00d6ff047aa0aa`，`completion.tsv` SHA256=`d7424046ffd55b34517e22d6776ccd13e72c5fb52c886dc6b2446f8b35aa6699`；launcher原生仅写`candidate_pids.tsv`，未写completion/manifest，故两者为本地机械receipt，不改变远端run。
- 本runner仅报告技术闭环与结构字段，未读取/解释性能，不启动后续run，retry=`NO`；root/Git报告待提交仅本报告终态修正。

## 6. 完整同fold结果

数值均为同一fold内`G-C`，单位为百分点（pp）；proxy AUROC差为0–1标度，FAR差为pp。

|Fold|clean Δoverall|clean Δmin-class|clean Δmin-RX|clean Δmin-day|三场景等权LEO Δoverall|最差LEO格|最差Δ|proxy ΔAUROC|proxy ΔFAR|clean门|LEO门|proxy门|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|
|F1|0.000|0.000|0.000|0.000|+0.184|clear/min-RX|-4.124|+0.00840|-8.250|通过|失败|通过|
|F2|0.000|0.000|0.000|0.000|+2.141|clear/min-class|0.000|-0.00118|-1.000|通过|通过|失败|
|F3|+0.188|+1.500|+1.887|+0.333|+3.876|rain/min-RX|+2.532|+0.00594|-1.500|通过|通过|通过|
|F4|-0.062|0.000|0.000|-0.034|+0.368|rain/min-class|-1.562|+0.02220|-9.000|通过|通过|通过|
|F5|-0.125|-0.250|-0.755|-0.256|+2.907|clear/min-day|-2.484|-0.02519|+3.250|通过|失败|失败|
|F6|+0.250|+1.500|+1.509|+0.489|+1.559|clear/min-RX|0.000|+0.00758|+11.750|通过|通过|失败|

18个LEO格的完整四指标差：

|Fold|场景|Δoverall|Δmin-class|Δmin-RX|Δmin-day|四floor门|
|---|---|---:|---:|---:|---:|---|
|F1|clear|-0.919|-1.389|-4.124|-0.932|失败|
|F1|low|+1.471|-0.781|+1.124|+1.923|通过|
|F1|rain|0.000|-1.562|-1.266|+0.446|通过|
|F2|clear|+0.735|0.000|+1.031|+0.856|通过|
|F2|low|+2.757|+1.562|+1.124|+3.846|通过|
|F2|rain|+2.930|+10.156|+3.797|+2.480|通过|
|F3|clear|+4.044|+12.500|+11.633|+5.280|通过|
|F3|low|+3.676|+13.281|+4.494|+3.411|通过|
|F3|rain|+3.906|+17.969|+2.532|+5.208|通过|
|F4|clear|-0.184|0.000|+2.062|-0.621|通过|
|F4|low|+1.287|+3.125|+1.124|+1.809|通过|
|F4|rain|0.000|-1.562|-1.266|+0.347|通过|
|F5|clear|+0.551|+3.472|+1.031|-2.484|失败|
|F5|low|+3.676|+4.688|+1.124|+3.846|通过|
|F5|rain|+4.492|+12.500|+1.316|+2.431|通过|
|F6|clear|+1.103|+2.083|0.000|+0.901|通过|
|F6|low|+2.206|+4.688|+1.124|+1.442|通过|
|F6|rain|+1.367|+3.906|+6.329|+1.389|通过|

proxy同fold原值：

|Fold|C AUROC|G AUROC|C FAR|G FAR|门|
|---|---:|---:|---:|---:|---|
|F1|0.57875|0.58715|47.50%|39.25%|通过|
|F2|0.52393|0.52275|88.50%|87.50%|失败：AUROC下降|
|F3|0.60648|0.61243|55.75%|54.25%|通过|
|F4|0.53348|0.55568|73.00%|64.00%|通过|
|F5|0.62354|0.59834|41.00%|44.25%|失败：两项均恶化|
|F6|0.57173|0.57931|25.50%|37.25%|失败：FAR上升|

## 7. 聚合与裁决

- 技术门：通过；42/42步骤、6/6 pair和全部绑定闭合。
- clean known门：6/6通过。六折均值为Δoverall`+0.042pp`、Δmin-class`+0.458pp`、Δmin-RX`+0.440pp`、Δmin-day`+0.089pp`。
- LEO平均方向：6/6 fold三场景等权overall均为正；18格等权Δoverall`+1.839pp`，Δmin-class`+4.702pp`、Δmin-RX`+1.844pp`、Δmin-day`+1.754pp`。
- LEO非补偿floor门：仅16/18格通过。F1-clear的min-RX由86.598%降至82.474%（`-4.124pp`）；F5-clear的min-day由80.124%降至77.640%（`-2.484pp`）。平均提升不得补偿这两个失败格。
- proxy门：仅3/6通过。六折平均ΔAUROC`+0.00296`、ΔFAR`-0.792pp`，但F2、F5、F6逐折失败，平均值不得补偿。
- 最终裁决：`REJECT_CB_SFCE_PERMANENT`。不调整lambda、gamma或场景采样，不重跑，不进入Phase1 advancement或Phase3。
- 科学含义：直接优化有标签LEO决策风险确实产生了广泛正向信号，且没有破坏clean总体稳定性；但它仍未消除跨fold的最差RX/最差day尾部风险，并且proxy拒识稳定性不足。因此它是有价值的负结果，而不是可晋级方法。
