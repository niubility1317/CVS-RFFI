# ERBT-IDR M2.6 TD-SRC256多表征screen正式实验报告

日期：2026-08-23
run ID：`erbt_idr_m26_td_src256_repr_screen_20260823_v1`
当前状态：`ANALYZED / SCREEN_NEGATIVE_NO_FULL125`
实现提交：`18f630dfc06c83591dd7810a2e60d4562b147150`
分支：`codex/m26-td-src256-20260823`

## 一、目标与证据边界

本实验以去RF32的D92 E0为唯一主基线，验证一个真正读取目标旧类support域偏移、但不允许query参与拟合的有界残差。除既有FFT96的正交包络/纹理分解外，本轮新增MGD96频谱幅度几何表征：低阶DCT去趋势残差、局部斜率和精确镜像不对称各32维。所有表征都由同一固定接收IQ的确定性FFT96派生，不增加物理K。

本报告已完成24行prediction、truth-last评分、分层分析和screen裁决。它是4个配对输入身份的机制筛选证据，不是完整125确认结果；不得把本报告中的绝对指标与历史full125绝对指标直接作性能排序。

## 二、冻结方法与矩阵

|arm|机制|
|---|---|
|`M24-D1-COMPILE-PARITY`|B0：去RF32 D92 E0|
|`M26-T1-G0-IDENTITY-DOMAIN-RESIDUAL`|T1：identity160目标域传输|
|`M26-T2-G0-SPECTRAL-DECOMP-RESIDUAL`|T2：Envelope32＋Ripple64稳健support残差|
|`M26-T3-G0-JOINT-TARGET-SHIFT-RESIDUAL`|T3：identity传输＋Envelope/Ripple|
|`M26-T4-G0-MAGNITUDE-GEOMETRY-RESIDUAL`|T4：MGD96稳健support残差|
|`M26-T5-G0-JOINT-MAGNITUDE-GEOMETRY-TARGET-SHIFT-RESIDUAL`|T5：identity传输＋MGD96传输|

screen固定为：

- receiver：`3-19`、`8-8`；
- method seed：`7282101`；
- 条件：K5/new20、K10/new5；
- 4个配对输入身份×6个arm＝24个方法行；
- 3个LEO场景/行，共72个场景单元；
- 强度网格：`{0,0.01,0.02,0.04}`；
- K1不在screen；未来full125中的K1必须逐位回退B0；
- query逐条在全部注册类上独立argmax。

完整125触发条件：T3或T5至少一者相对B0达到`ΔH>=0.002`、`N_help>N_harm`，且`min-old`和`min-new`下降均不超过0.005。达到后使用新run ID运行5 receiver×5 seed×5 K/new×6 arm＝750行完整矩阵；未达到则以screen负结果闭环，不启动局部或伪full125。

## 三、协议与Phase1锚点

- `protocol_schema=p2_min_v1`；
- 复用`phase2_data_status=VALIDATED_ONCE`且匹配原`capsule_id/split_id`；
- target域状态只由六个目标旧类support与Phase1六类聚合锚点估计；
- 新类support、query、query truth均不进入域状态；
- Phase1锚点只保存6×identity160和6×FFT96的int8中心及FP16尺度；
- 锚点绑定真实checkpoint、D19类映射和量化内容component ID；该component ID进入候选锁；
- 混合Phase1导出只筛选`dataset_role=source`，target行使用数为0。

正式输入：

- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- source导出：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz`
- D19类绑定：release内`analysis/d19_adv3b02_class_binding_20260717.json`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`

## 四、环境与不可覆盖路径

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m26_td_src256_repr_screen_20260823_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：release内`code`
- prediction：CPU，`max-workers=2`；不占用训练GPU。

## 五、预登记命令

锚点构建：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/build_m26_phase1_spectral_anchor.py \
  --source-npz /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_publication_adv3b02_fft96_singleview_20260714/leo_clear_weak.npz \
  --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor.npz \
  --audit-output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor_audit.json \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --class-binding-json ../analysis/d19_adv3b02_class_binding_20260717.json
```

prediction：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m26_td_src256_matrix.py \
  --run-id erbt_idr_m26_td_src256_repr_screen_20260823_v1 --matrix-kind screen \
  --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features \
  --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features \
  --source-anchor /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/control/m26_source_anchor.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions \
  --device cpu --max-workers 2
```

truth-last评分：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m26_td_src256_matrix.py \
  --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions/matrix_index.json \
  --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars \
  --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/scores \
  --bootstrap-repeats 2000
```

汇总：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m26_td_src256_matrix.py \
  --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/predictions \
  --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/scores \
  --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m26_td_src256_repr_screen_20260823_v1/results_summary.json
```

## 六、实验前证据与停止规则

- M2.6聚焦测试：18项通过；
- M2.4/M2.5/M2.6相邻回归：25项通过；
- `py_compile`和`git diff --check`通过；
- 独立审查首次P0=0、P1=2；最小修复后定点复审P0=0、P1=0、READY；
- 实现提交已推送且远端OID等于本地HEAD。
- 正式release提交：`8546ae8312dce19c10a2486a753f8be9839e1dde`；发布前仅存在未纳入提交的本地pytest临时目录，不影响归档内容；
- 2026-08-23T20:30:17+08:00直连N607预检通过：正式输入齐全，release/run/log目标均不存在；GPU0–2存在其他任务负载，GPU3–7空闲，本轮prediction固定CPU执行；
- 唯一release归档由上述提交生成，本地与远端映射为`E:\type10-7\local_artifacts\erbt_idr_m26_td_src256_repr_screen_20260823_v1.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m26_td_src256_repr_screen_20260823_v1.tar.gz`，单次传输SHA-256均为`1bd82ffc9d9c80a679b6bc73b1c008d4b3f1ecd320c018f2de4d4105de41f60b`；
- 远端release一次编译通过；Phase1锚点component ID为`236823c57210de9a58a47ee1868d27a907368864f8ad9f710c2eda0c51258da5`，只使用2400条source行，target/query使用数均为0，持久状态1560B；
- 真实checkpoint无query smoke通过：checkpoint SHA-256匹配，missing/unexpected均为0，`tx_logits=[2,6]`、`z_id=[2,160]`且有限，`query_input_count=0`。
- prediction于2026-08-23T20:33:59+08:00启动，唯一父PID为`1718031`；首次读回确认CWD、cmdline、run root与预登记一致，两个CPU工作子进程归属明确，启动时收据0/24且无异常指纹。

只允许因协议/query泄漏、错误矩阵/checkout、输出碰撞、进程归属不清、确定性重复异常、prediction不闭合或scorer连接错误停止；低性能不得作为技术停止理由。

预期artifact：24份`row_execution_receipt.json`、`matrix_index.json`且状态为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、B0及T1–T5各4行、24份same-row/four-state score、20份paired-vs-B0结果、`scored_matrix_index.json`和`results_summary.json`。最终报告必须覆盖总体、K/new、receiver、seed、scene、四状态、old/new、class、margin、中心角距、help/harm、F_within/F_std、域偏移、LOO可靠度、强度、门控、回退和资源。

## 七、执行闭环与证据完整性

本轮实验没有发生技术失败，最终闭环如下：

- prediction完成24/24行，4个配对输入身份均恰好包含B0及T1–T5六臂；
- `matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`，`row_count=24`，`paired_input_identity_count=4`，三场景单元共72个；
- 全部prediction阶段的query输入计数和query拟合计数均为0；
- B0注册前/注册后prediction disagreement均为0，去RF32主基线保持确定性；
- prediction完整后才运行独立truth-last scorer，最终产生89份score JSON且`scored_matrix_index.status=PASS`；
- 远端原始24份`.cvspred`约45.5MB，保留在不可覆盖run root；Git只发布matrix index、24份receipt、score、控制证据、日志和机器可读汇总；
- 汇总器曾继承M2.4 full125的证据边界文字。该问题只影响标签，不影响任何prediction、score或数值数组；已增加回归测试并将正式汇总修正为`matrix_kind=screen`，远端原始汇总另存为`results_summary_remote_raw.json`。

## 八、总体结果与预登记screen裁决

下表均为本轮4440条注册后query的query加权结果。B0是同row去RF32 D92 E0基线。

|arm|`A_o_pre`|`A_o_post`|`A_n`|H|F|`min-old`|`min-new`|H相对B0|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B0：去RF32 D92 E0|0.742005|0.649362|0.588896|0.610486|0.092643|0.268919|0.251577|0|
|T1：identity目标域残差|0.746134|0.644482|0.591937|0.609557|0.101652|0.273874|0.259009|−0.000929|
|T2：Envelope/Ripple残差|0.741179|0.649212|0.586644|0.609271|0.091967|0.263063|0.251577|−0.001215|
|T3：identity＋Envelope/Ripple|0.744895|0.644482|0.591937|0.609557|0.100413|0.273874|0.259009|−0.000929|
|T4：MGD96幅度几何残差|0.742005|0.649775|0.588896|0.610728|0.092230|0.268919|0.251577|+0.000242|
|T5：identity＋MGD96|0.744895|0.645458|0.591351|0.609757|0.099437|0.273874|0.259009|−0.000729|

预登记只允许T3或T5触发完整125。T3的`ΔH=−0.000929`、help/harm=17/13；T5的`ΔH=−0.000729`、help/harm=15/12。两者的`min-old`和`min-new`均相对B0提高，地板保护条件通过，但最重要的`ΔH>=0.002`条件失败。T4虽是六臂中H最高者，但增益仅0.000242，也远低于0.002，且它不属于预登记的联合目标域候选。

因此正式裁决为：

```text
SCREEN_NEGATIVE_NO_FULL125
```

这不是实验执行失败，而是一个有效的科学负结果。按预登记和最小实验工作流，不启动750行full125，也不把局部screen包装为完整125。

## 九、same-row、help/harm与条件稳定性

### 9.1四个配对输入身份

|arm|row等权H差均值|最小|最大|正/负/平|
|---|---:|---:|---:|---:|
|T1|−0.000326|−0.003625|+0.005792|1/2/1|
|T2|−0.000708|−0.003915|+0.001625|1/2/1|
|T3|−0.000326|−0.003625|+0.005792|1/2/1|
|T4|+0.000406|0|+0.001625|1/0/3|
|T5|−0.000184|−0.003471|+0.005792|1/2/1|

|arm|`N_help`|`N_harm`|注册后总体accuracy差|McNemar显著row数|
|---|---:|---:|---:|---:|
|T1|17|13|+0.000901|0|
|T2|2|8|−0.001351|1|
|T3|17|13|+0.000901|0|
|T4|4|3|+0.000225|0|
|T5|15|12|+0.000676|0|

T1/T3的总体accuracy略升但H下降，说明新增正确预测没有同时改善old/new平衡。T2出现2/8的明显负净翻转，且一个row的McNemar p=0.015625。T4只翻转7条query，不能据此声称稳定增益。

### 9.2K/new条件

|条件|B0 H|T1|T2|T3|T4|T5|
|---|---:|---:|---:|---:|---:|---:|
|K5/new20|0.564465|0.562652|0.562507|0.562652|0.564465|0.562936|
|K10/new5|0.719264|0.720425|0.719805|0.720425|0.720077|0.720425|

所有目标域联合候选都只在K10/new5出现小幅正信号，却在更困难的K5/new20退化。域偏移估计并非完全无信息，但它对新类数量和support规模敏感，尚未形成可跨条件使用的稳健中心。

### 9.3receiver与场景

|receiver|B0 H|T1|T2|T3|T4|T5|
|---|---:|---:|---:|---:|---:|---:|
|3-19|0.526178|0.525146|0.526016|0.525146|0.526178|0.525146|
|8-8|0.694794|0.693968|0.692526|0.693968|0.695277|0.694367|

|scene|B0 H|T1|T2|T3|T4|T5|
|---|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.670156|0.666554|0.666425|0.666554|0.670156|0.667546|
|`leo_low_elev_weak`|0.581155|0.581549|0.580517|0.581549|0.581155|0.581155|
|`leo_rain_weak`|0.580147|0.580569|0.580872|0.580569|0.580872|0.580569|

T4的微小总体增益全部来自receiver8-8及部分rain切片，在receiver3-19和clear场景没有增益。单seed设计不能给出跨seed稳定性结论；这一限制属于预登记screen证据边界，而不是缺失实验行。

## 十、四状态、遗忘与old/new关系

|arm|`DA0_REG0`旧类|`DA1_REG0`旧类|`DA0_REG1`旧类/新类/H|`DA1_REG1`旧类/新类/H|
|---|---:|---:|---:|---:|
|B0|0.709234|0.742005|0.548198/0.430563/0.478404|0.649362/0.588896/0.610486|
|T1|0.709234|0.746134|0.548198/0.430563/0.478404|0.644482/0.591937/0.609557|
|T2|0.709234|0.741179|0.548198/0.430563/0.478404|0.649212/0.586644/0.609271|
|T3|0.709234|0.744895|0.548198/0.430563/0.478404|0.644482/0.591937/0.609557|
|T4|0.709234|0.742005|0.548198/0.430563/0.478404|0.649775/0.588896/0.610728|
|T5|0.709234|0.744895|0.548198/0.430563/0.478404|0.645458/0.591351/0.609757|

所有候选都严格保持`DA0`两态，说明变化只来自目标域路径。T1/T3/T5提高`DA1_REG0`旧类，但注册后旧类反而低于B0；域偏移对无新类竞争和有新类竞争的作用方向不一致。这是当前方法最关键的失败机制。

|arm|`F_within`|`F_std`|
|---|---:|---:|
|B0|0.092643|0.092643|
|T1|0.101652|0.097523|
|T2|0.091967|0.092793|
|T3|0.100413|0.097523|
|T4|0.092230|0.092230|
|T5|0.099437|0.096547|

T1/T3/T5的`F_within`升高部分来自注册前准确率提高，但标准化遗忘仍高于B0，因此不能把它解释为更强的抗遗忘能力。T2/T4的遗忘略低，却没有足够H增益。

## 十一、margin、中心角距与类别诊断

|arm|margin均值/中位数|中心角距均值/中位数|
|---|---:|---:|
|B0|0.566556/0.295723|24.968851°/24.884703°|
|T1|0.566828/0.295723|63.445514°/65.894253°|
|T2|0.566602/0.295723|21.682050°/22.023991°|
|T3|0.566818/0.295723|47.011815°/47.517056°|
|T4|0.566557/0.295723|31.618141°/30.802833°|
|T5|0.566791/0.295723|50.998998°/51.940035°|

所有候选都保持了B0的margin中位数，没有重现M2.4 G1–G4的margin塌缩；有界主判决保护是有效的。然而T1/T3/T5把中心几何大幅旋转到47°–63°，H仍下降，证明“中心明显改变”不等于“old/new竞争得到校准”。

按26个类别汇总，T1/T3为6类上升、6类下降、14类持平；T2为1/4/21；T4为4/2/20；T5为6/6/14。T1/T3最差类别包括`6-15`（−0.0208）、`16-19`（−0.0083）和`2-5`（−0.0083），最大正向类别为`1-18`（+0.0500）。T4变化范围更小，但仍有`16-19`下降0.0167。当前变体没有获得类别一致性。

## 十二、目标域状态、LOO可靠度与门控失配

目标域状态确实被构造且在注册前后保持同一digest；不存在“代码没有用到新状态”或query参与拟合的问题。主要汇总如下：

- identity可靠度0.978245，域偏移范数0.350000；
- Envelope可靠度0.484388，偏移范数0.214320；
- MGD96可靠度0.500000，偏移范数0.252930；
- identity/Envelope/MGD96的support-LOO增益均值分别为0.065683/0.012062/0.051071；
- 三种LOO正向比例分别为0.786036/0.802553/0.641892；
- T1/T3/T5有61.71%的scene级fit回退零强度，T4回退66.67%；
- 最终实际修改query的比例只有T1/T3/T5的9.32%和T4的8.42%，最大logit增量约0.0133–0.0153。

这组结果否定了“只要support留一重构更好，就可以安全地把域偏移直接加到分类分数”的假设。LOO度量只验证类内中心或重构，不验证注册后的old/new类间竞争方向；因此即使identity可靠度很高、动作幅度很小，仍可能提高注册前旧类而降低注册后H。

T1与T3的注册后预测和核心指标完全相同，说明在当前选择规则下Envelope/Ripple没有贡献可观测的额外决策信息。MGD96比CEP96更安全，但信号太弱且局部化，尚不能视为比FFT96更有效的正式表征。

## 十三、资源代价

|arm|state bytes|注册时间/row|query head MAC|batch head延迟/row|MAC等价值|
|---|---:|---:|---:|---:|---:|
|B0|13921B|10.40ms|5514|1.45ms|7848960|
|T1|72926B|46004ms|7499|18.72ms|10474430|
|T2|72926B|47231ms|6705|11.75ms|9424242|
|T3|72926B|45302ms|8690|11.80ms|12049712|
|T4|72926B|45981ms|6548|13.39ms|9320640|
|T5|72926B|46333ms|8690|18.86ms|12049712|

Phase1量化锚点本身仅1560B，真正的资源瓶颈来自逐row support-LOO拟合和候选状态布局。与B0相比，当前M2.6约5.2倍状态、45秒级注册和8–13倍batch head延迟，却没有达到科学阈值。即使未来获得性能信号，也必须先向量化LOO并压缩状态；当前版本不能作为部署候选。

## 十四、与完整125历史结果的严格对比

|实验|证据规模|方法|H|相对同run去RF32 D92 E0|
|---|---|---|---:|---:|
|D92 E0历史主基线|full125|去RF32 B0|0.537558|0|
|M2.5完整确认|full125|B3稳定双原型残差|0.539228|+0.001669|
|M2.6当前screen|4 identity|B0|0.610486|0|
|M2.6当前screen|4 identity|T4 MGD96|0.610728|+0.000242|
|M2.6当前screen|4 identity|T3目标域联合残差|0.609557|−0.000929|
|M2.6当前screen|4 identity|T5 MGD96目标域联合残差|0.609757|−0.000729|

M2.6的绝对H高于历史full125是因为screen只包含两个receiver、一个seed及K5/new20、K10/new5，不能解释为算法提升。唯一合法比较是M2.6同row候选减同row B0；在该口径下没有候选超过M2.5完整125已经得到的+0.001669，也没有候选达到本轮+0.002的晋级阈值。

因此当前模块二的证据层级保持为：

1. 去RF32 D92 E0仍是部署默认和所有后续实验的唯一主基线；
2. M2.5 B3仍是现有完整125中最佳的科学性能分支，但资源代价较高；
3. M2.6验证了目标域偏移可在协议内构造和使用，同时否定了直接独立叠加identity/CEP/MGD目标偏移分数的当前形式；
4. M2.6不晋级、不替代M2.5、不产生full125性能声明。

## 十五、下一轮优化设计建议

下一候选不应继续增加一个独立“域中心分类头”，而应改成`M2.7 B3-CONDITIONED-SPECTRAL-VETO`：

1. 主判决固定为去RF32 B0，性能分支固定为已经通过full125的M2.5 B3；
2. 目标域偏移只调制B3残差的符号、强度或是否启用，不直接生成新的类别logit；
3. support校准目标从类内LOO重构改为“留一old-class竞争损失”：只有当域状态降低真实旧类对最危险竞争类的相对logit，并且旧类风险上界不恶化时才允许动作；
4. MGD96只保留为共识否决器：B3与MGD方向一致且风险预算通过时启用，否则逐query回退B3或B0；
5. 继续去RF32，所有新增视图只能由固定接收IQ确定性生成，不增加K，不允许query更新；
6. 新表征优先探索低秩16–32维复频域侧信息，而不是再堆一个96维幅度头：相邻bin相位增量、相位线性残差、镜像相位相干、归一化倒谱导数；
7. 对target shift施加class-shared、zero-sum和receiver条件化约束，避免整体旋转中心却破坏old/new标尺；
8. 在下一次screen前将注册过程向量化，目标从45秒/row降至100ms量级；资源优化不改变科学阈值。

建议下一次最小矩阵仅包含B0、B3、B3＋MGD共识否决、B3＋phase-coherence否决四臂，沿用同一4 identity screen。只有候选达到`ΔH>=0.002`、help>harm、地板下降不超过0.005，才启动完整125。该方案把本轮得到的微弱MGD安全信号用于“是否相信B3”，而不是重复已经失败的独立分类残差。

## 十六、发布物与最终结论

- 正式报告：`automation_reports/CV-SincNet/erbt_idr_m26_td_src256_repr_screen_20260823_v1/report.md`；
- 机器可读正式汇总：同目录`results_summary.json`；
- 远端原始汇总：同目录`results_summary_remote_raw.json`；
- prediction闭合、score、控制证据和日志：同目录`evidence/`；
- 实现追踪：`docs/ERBT_IDR_M26_TD_SRC256_TRACE_20260823.md`；
- 历史总对比：`docs/D92_E0_ALL_ABLATION_EXPERIMENTS_REPORT_20260819.md`。

最终结论：M2.6工程与协议验证通过，科学screen未通过。目标域域偏移不是不存在，而是当前“类内可靠度→独立类别分数”的作用方式错误；CEP96无独立收益，MGD96只有低幅、局部安全信号。模块二下一步应保留去RF32 B0和M2.5 B3，把目标域与新频域表征降级为有界共识/否决信息，直接优化注册后的old/new竞争，而不是再次替换稳健中心。
