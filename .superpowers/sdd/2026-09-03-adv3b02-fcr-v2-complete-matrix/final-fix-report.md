# FCR-V2最终审查修复报告

日期：2026-09-03

基线提交：`5203e986debe22f2a408e388b89c7fcdbd64c480`

设计权威：`docs/superpowers/specs/2026-09-03-adv3b02-fcr-v2-complete-matrix-design.md`

精确finding清单：`.superpowers/sdd/2026-09-03-adv3b02-fcr-v2-complete-matrix/final-review-report.md`

最终判定：`DONE_WITH_CONCERNS`

## 修复边界

- 未改变ManySig数据协议、`seed=392005`、`epochs=200`、固定checkpoint、final-only和truth-last边界。
- `cross_decode`继续使用source内容及target fingerprint+nuisance，没有改回source fingerprint。
- checkpoint重建使用`load_state_dict(...,strict=True)`，没有忽略missing/unexpected key。
- 科学metadata只从ManySig索引可验证语义构造。索引不能证明`common_preamble_id`和`excitation_bin`时保留空值/-1，fingerprint机制必须报告`MECHANISM_NOT_ACTIVATED`，不伪造pair或覆盖率。
- 两个既有`local_artifacts`目录未修改、未删除、未stage。

## RED证据

修复前先用真实入口聚焦测试复现：

1. launcher的实际`COMMON_ARGS`进入真实`code/train.py`parser后失败，错误包含`unrecognized arguments: true --phase1_source_role_protocol ... --phase1_external_final_eval true`。
2. V2 checkpoint进入正式`build_exact_ssdg_model_from_checkpoint`路径时因重建层遗漏`fcr_version`而按V1建模，prediction前报`decoder_mode must be 'control' or 'full_physics'`。
3. 新增审查回归首轮因尚无具名`CRRA_NUISANCE_UNITS`而在收集期失败，证明eta字段/单位契约在实现前不存在。

## 逐finding修复追踪

|ID|修复文件|实现结果|聚焦验证|剩余风险|
|---|---|---|---|---|
|P0-1|`code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`；`code/tests/test_phase1_fcr_v2_complete_launcher.py`|删除parser不存在的5个参数；`store_true`参数改为无值形式；真实preflight对C1/M6均调用`build_train_command`生成完整正式argv并进入`--fcr_config_dry_run`。不可覆盖output root、两波GPU映射和truth-last顺序保持不变。|C1/M6完整正式argv均由真实parser返回0；launcher真实Git Bash dry-run通过。|本轮未执行N607远端launcher smoke；它仍属于后续最小发布流程。|
|P0-2|`code/post_stage_common.py`；`code/cvsrffi/checkpoint_loading.py`；`code/cvsrffi/eval.py`；`code/scripts/predict_phase1_truth_last.py`；`code/tests/test_phase1_fcr_v2_final_review_fixes.py`|checkpoint参数中的`fcr_version`显式进入模型工厂；V1/V2输出schema分别严格校验；正式predictor对V2只走`forward_identity_only`且不运行decoder；严格加载改为`strict=True`。|C2、S4、M6各自保存parser完整的V2 `final.pth`，经正式predictor严格重建、四场景生成prediction，再经独立scorer闭合；missing/unexpected均为0，decoder调用为0。|使用最小不含truth的opaque输入fixture代替168000条正式包，模型重建、predictor循环和scorer均为生产入口；正式N607 checkpoint身份仍待发布smoke。|
|P1-1|`code/baseline_origin_sat_view.py`；`code/cvsrffi/phase1_fcr_types.py`；`code/cvsrffi/phase1_fcr_v2_metadata.py`；`code/cvsrffi/phase1_fcr_v2_factors.py`；`code/train.py`；相关contract/审查测试|统一9维`snr_db,cfo_hz,residual_cfo_hz,fD_hz,pl_db,K_db,theta_deg,h_km,state`字段、单位、尺度和顺序；V2新增独立eta head；错序/错单位/错宽度失败关闭；clean行全维mask为false，不再把零值当科学标签。|具名schema、单位、尺度、clean mask、错误单位、严格shape及真实objective回归通过。|eta是可学习观测目标，不等同于物理解码器内部参数；正式误差只可由训练期有效mask统计解释。|
|P1-2|`code/dataset_wisig.py`；`code/cvsrffi/phase1_fcr_v2_pairing.py`；`code/cvsrffi/phase1_fcr_interventions.py`；`code/train.py`；审查测试|ManySig索引可验证地提供physical/content record及`eq:0/1`链路条件；无法证明的preamble/excitation保留空值/-1；删除训练入口合成preamble/excitation/link的fallback；三轴PairBuilder逐条件严格校验，U_s不保留可逆TX身份。|真实索引语义helper、缺metadata失败、nuisance/content/fingerprint条件及零pair未激活诊断回归通过。|当前ManySig索引不能验证跨TX共同preamble/excitation，因此fingerprint pair必为0；M4及M6的fingerprint依赖部分在正式数据上必须如实未激活。content pair是否在某个训练batch出现由实际PairBuilder计数决定，不静态保证。|
|P1-3|`code/baseline_origin_sat_view.py`；`code/tests/test_phase1_fcr_v2_final_review_fixes.py`|scenario、应用判定、全部eta参数和IQ随机量均由`seed+epoch+physical_sample_id+view_type`的无状态generator生成，不读取batch位置或前序随机状态。|同一physical sample在重排、不同`batch_idx`下的scenario、eta和IQ逐样本完全一致。|旧的无metadata兼容路径仍保留batch级generator；正式V2入口强制提供physical sample metadata并进入无状态路径。|
|P1-4|`code/cvsrffi/phase1_fcr_v2_losses.py`；`code/cvsrffi/phase1_fcr_v2_schedule.py`；`code/train.py`；loss/schedule/integration测试|shared-f改为clean teacher stop-gradient→LEO student；加入跨step持久`LossMagnitudeEMA`；tail使用source标注batch的per-class CVaR；M1-M6均继承true swap；E161-200保留生成项0.10及self/shared 0.25。训练证据记录实际有效权重、非零loss和梯度。|teacher无梯度/student有梯度、EMA跨step更新、劣势类tail非零、M1-M6在E120/E180含swap，以及真实M6反传均通过。|梯度证据为E200前4个成功batch的保守抽样；抽样中无pair或梯度为0时机制会被标记未激活，不会上调为静态active。|
|P1-5|`code/cvsrffi/phase1_fcr_v2_losses.py`；`code/train.py`；loss/integration测试|M2的decode→waveform→`forward_identity_only`→三轴恢复不再注入外部fingerprint latent，并闭合content/fingerprint/nuisance；M3改为`relu(m_rel-(drop-full)/(full+eps))`，full作为stop-gradient基准。|M2真实decode/re-encode图产生非零cycle损失和梯度；M3不足margin时梯度推动drop误差增大，超过margin时loss为0。|真实nuisance pair由clean→LEO有效eta决定；没有有效pair时M2/M3严格未激活。|
|P1-6|`code/train.py`；`code/tests/test_phase1_fcr_v2_training_integration.py`|M4只从生成波形重编码；成熟ADV3B02 classifier为冻结副本；身份CE按destination class先分组再等权平均；同时保持source内容、destination fingerprint和destination nuisance。|M4真实decode/re-encode图、非零transplant损失/梯度、冻结classifier参数及M6继承图均通过。|当前正式ManySig缺少可验证fingerprint pair，因此该图只在显式有效科学metadata fixture中验证；正式数据不得声明激活。|
|P1-7|`code/cvsrffi/phase1_fcr_v2_losses.py`；`code/train.py`；审查/integration测试|M5实际计算clean和LEO复数物理Gram gate，并约束response surface二阶平滑；M6实际执行fingerprint/content/nuisance三轴decode→re-encode，different-TX fingerprint使用margin separation而非正对齐。|等幅不同相位Gram可区分；response扰动提高smoothness loss；M5三项物理量和梯度非零；M6三轴loss及factor梯度均非零。|完整M6需要三轴pair同时存在；正式ManySig缺fingerprint语义时整体factor机制必须未激活，不能用content/nuisance部分冒充完整三轴覆盖。|
|P1-8|`code/cvsrffi/phase1_fcr_v2_diagnostics.py`；`code/train.py`；diagnostics/审查测试|diagnostics消费成功optimizer step后累积的三轴PairBuilder真实count/opportunity、eta有效计数/误差、有效权重、非零loss step和相对CE梯度；静态配置与实际激活分栏；无证据写`MECHANISM_NOT_ACTIVATED`；V2诊断收集/写出异常一律抛出技术失败，即使truth-last defer也不能正常返回。|零pair/零梯度不激活、真实pair覆盖计算、eta实际计数、注入写失败向外传播，以及diagnostics-before-defer顺序均通过。|诊断不再用matched clean/LEO行数伪造pair覆盖；未获得训练期证据时显示N/A/未激活。|

## GREEN验证记录

全部命令在`ssr-gpu`环境串行执行。

### 1.语法及聚焦回归

```text
python -m py_compile code/train.py code/baseline_origin_sat_view.py code/dataset_wisig.py code/scripts/predict_phase1_truth_last.py
python -m pytest code/tests/test_phase1_fcr_v2_diagnostics.py code/tests/test_phase1_fcr_v2_training_integration.py code/tests/test_phase1_fcr_v2_final_review_fixes.py code/tests/test_phase1_fcr_v2_complete_launcher.py -q
```

结果：退出码0，49项通过；仅有既有`torch.cuda.amp.autocast`弃用warning。

### 2.全部Phase1 FCR回归

```text
python -m pytest code/tests -k phase1_fcr --disable-warnings
```

结果：退出码0，173项通过、1405项deselected、25条warning，用时36.10秒。

### 3.launcher/predictor/scorer闭环

```text
python -m pytest code/tests/test_phase1_fcr_v2_complete_launcher.py code/tests/test_phase1_fcr_v2_final_review_fixes.py code/tests/test_exact_ssdg_checkpoint_loading.py code/tests/test_phase1_fcr_r1r8_s392005_release.py --disable-warnings
```

结果：退出码0，26项通过、2条warning，用时19.35秒。该组包含真实Git Bash launcher dry-run、C1/M6完整正式argv进入真实parser、C2/S4/M6 V2严格checkpoint→正式predictor→独立scorer闭环。

### 4.Git范围检查

```text
git diff --check
git status -sb
```

结果：无空白错误；两个无关`local_artifacts`目录仍为未跟踪且不在交付范围。行尾仅报告仓库既有Windows LF/CRLF提示。

## 最终剩余风险与结论

工程层面的2项P0和8项P1均已按原finding定点修复并回归通过。唯一实质关注项是数据证据边界：当前ManySig索引不能证明跨TX共同preamble和excitation bin，因此不能真实构造fingerprint pair。实现不会填占位值，也不会宣称M4/M6已获得正式科学覆盖；正式训练会继续到E200并在diagnostics中严格写`MECHANISM_NOT_ACTIVATED`。这符合本轮“无法构造时严格未激活”的要求，但意味着在补充可验证索引metadata前，M4和完整M6不能作为已激活机制解释。

本轮未运行N607、未同步release、未启动矩阵，也未核验正式`seed=392005/E200/ADV3B02_CORE90_SOFT_E200`远端checkpoint；这些属于后续Task8最小发布流程，不能由本地测试冒充。

交付提交以本报告所在Git提交为准；push后由最终回执独立比较本地`HEAD`与远端分支OID。
