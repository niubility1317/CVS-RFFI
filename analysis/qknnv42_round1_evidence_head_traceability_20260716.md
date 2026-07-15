# qKNNv42第1轮EvidenceNorm注册头追踪表

日期：2026-07-16
状态：Round1本地算法诊断完成；1/3轮。N607基线矩阵已完成，本轮没有远端同步或启动。

## 1.轮次与复盘口径

本轮是“三轮算法探索→强制复盘”新计数周期的第1轮。历史严格运行链修复、协议握手修复、K1/K10基线和v11/v12/v13矩阵本身不计入新算法探索轮。计划固定为：

1. Round1：零梯度、类对称`EvidenceNorm`注册头；
2. Round2：在Round1头上叠加`joint_proj.0`的JP-R4 support-only稀疏更新；
3. Round3：在Round1头上叠加`id_gate.0+joint_proj.0`的JG-R8 LOPO更新；
4. Round3完成后停止第4轮启动，重新阅读目标、`项目.md`、conversation index、三轮完整日志与报告，并记录经验、淘汰路线和下一决策。

## 2.需求追踪矩阵

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|R1-01|用户目标；`项目.md`7.1、9.3|输入仅允许密封`leo_*_weak`注册support；不接收clean或clean-derived signal|`code/cvsrffi/phase2_symmetric_head.py`、`code/cvsrffi/stage2_predictor_runtime.py`|verified|真实sealed v11 package本地预测；pre-open/package audit PASS|head API不接受dataset路径或clean统计拟合入口|
|R1-02|`项目.md`7.2|逐样本面对所有注册类；禁止query真值、old/new角色、真实批次类数、类别配额及全局分配|同上|verified|类置换等变测试；prediction manifest五项access均为false|允许已注册类别数和support标签|
|R1-03|用户“域适应+新类注册同等重要”；`项目.md`9.3、10.3.1|同一run同时输出注册前old与注册后old/new/H/最低旧类/遗忘|运行时、独立scorer、活动报告|pending|20新类单cell已验证；5/10新类与独立矩阵待完成|只完成support loss或old-only不得晋升|
|R1-04|用户禁止角色Oracle与类别配额|所有注册类使用完全相同的原型、负证据、收缩和评分公式|`phase2_symmetric_head.py`|verified|类别轴置换等变、精确schema和交互项fail-closed测试|没有old/new专属bias或threshold|
|R1-05|用户要求K1正收益；`项目.md`10.3.1|K≥2采用leave-one-physical-support-out；K1仅采用leave-one-view-out并强收缩|`phase2_symmetric_head.py`|verified|K1/K2构造测试与`fold_mode`诊断|K1性能正收益尚未验证，不能声称物理样本LOPO|
|R1-06|极轻型资源门槛|0可训练参数、0epoch、0optimizer step、0额外backbone forward；每类仅2个FP16校准量|head状态、资源回执、报告|verified|256维26/6类精确字节账；256KB组合越界fail-closed|EvidenceNorm-only为104/24/128B；部署完整head为14,820B|
|R1-07|当前v1严格基线不可回归|`cvs.phase2.symmetric_locked_head.v1`行为逐位兼容|head、runtime、现有测试|verified|reference parity、v1资源keyset不变、sealed pipeline E2E PASS|新模式使用独立v2 schema，v1路径不变|
|R1-08|设计可达性|新head必须接入`build_formal_support_state`和`predict_formal_scenario_streams`，不能只存在于未调用模块|runtime validator与support state|verified|runtime集成测试和真实sealed package本地诊断|正式独立矩阵仍待后续候选锁|
|R1-09|自适应多View是重点但当前先隔离adapt|Round1/2/3保持同一TTA策略，避免把View变化归因于adapt/head|活动报告、候选锁|verified-with-boundary|raw cosine gate单元测试；1560行中仅1行view差异|逻辑已隔离；单行差异位于不同GPU数值阈值边界，正式同机复核|
|R1-10|MRIOR强制对比|锁定候选最终需与matched MRIOR-SDA比较性能与参数/step/时延/显存/状态/View|最终矩阵与报告|deferred|独立确认矩阵|Round1单cell不支持显著性声明|
|R1-11|三轮强制复盘|Round3结束后写回目标、协议、历史路线、完整日志和下一决策|活动报告|deferred|复盘章节|第4轮启动前硬门|

## 3.Round1算法定义

输入为`V=3K`个注册support观测、`C`个已注册类、`D`维ADV3B02+FFT特征：

```text
support_observations: [3K,C,D]
physical_shots_per_class: K
locked hyperparameters: negative_quantile, prior_shots, scale_floor, inverse_scale_cap
```

先按v1同一规则得到单位化原型`p_c`。对每个类使用同一公式：

```text
g_c = median_i cosine(z_ci, p_c^(-physical i))   # K>=2
g_c = median_i cosine(z_ci, p_c^(-view i))       # K=1
b_c = Q_q^higher {cosine(z_j,p_c): y_j != c}
r_c = g_c-b_c
w_K = K/(K+prior_shots)
b_tilde_c = w_K*b_c + (1-w_K)*median_c(b_c)
r_tilde_c = w_K*r_c + (1-w_K)*median_c(r_c)
d_tilde_c = max(r_tilde_c, scale_floor, 1/inverse_scale_cap)
score_c(q) = (cosine(z_q,p_c)-b_tilde_c)/d_tilde_c
```

输出为原型、每类FP16`b_tilde_c/d_tilde_c`、类对称预测分数和仅用于审计的闭式求解诊断。该方法是`EVAL_ONLY_CLOSED_FORM_ADAPTATION`，没有训练epoch或loss trace；必须保存求解诊断。

`inverse_scale_cap`用于限制`1/d_tilde_c`，防止`g_c<=b_c`时小gap被放大成新的prototype hub。Round1自适应TTA的1→3→5触发继续读取EvidenceNorm之前的raw cosine分数，EvidenceNorm分数只用于最终多View融合和类别预测；因此Round1不会把head变化与View触发率变化混在一起。

## 4.当前证据与预期边界

- v11诊断显示K10注册前old为76.94%，注册后59.44%，seen-new为35.50%，遗忘17.50pp；63/360个原本正确的旧类query全部被新增原型截获。
- v11 shard0的48个cell显示5/10/20新类、K1/5/10/20下均存在显著注册遗忘，所有聚合组合的旧类全局floor均为0。
- 因当前`use_alignment=false`，注册前后旧类原型本身一致；遗忘来自新增类竞争和hubness，不是“全局alignment重拟合”。Round1直接针对这一机制。
- Round1理论上能减少old→new截获和new→wrong-new hub错误，但不改变ADV3B02表征上限；它不能在实验前被宣称可单独达到92%旧类目标。

## 5.未闭合高风险项

最高风险是K1只有1个物理support，三种LEO场景仍是相关观测。若强收缩后EvidenceNorm只能减少遗忘、不能让K1注册后超过直接ADV3B02至少2pp，则Round2必须依靠JP-R4的support-only边界学习提升表征，而不能继续放大类bias或引入old/new专属规则。

## 6.真实20新类单cell诊断

固定输入为v11同一ADV3B02/effective8密封package：receiver=`20-1`、seed=`713101`、K=10、20个真实seen-new TX、3个`LEO_weak`场景。先生成1560行truth-free预测并冻结NPZ，再由独立scorer加入标签；该结果是`NON_LAUNCH_DIAGNOSTIC`，不是独立确认或部署成功。

|场景|注册前old|注册后old|direct old|seen-new|H|最低旧类|遗忘|平均View|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|80.83%|68.33%|74.17%|42.00%|52.02%|35.00%|12.50pp|2.654|
|`leo_low_elev_weak`|77.50%|60.83%|66.67%|33.00%|42.79%|20.00%|16.67pp|2.758|
|`leo_rain_weak`|77.50%|60.83%|70.00%|36.00%|45.23%|30.00%|16.67pp|2.769|
|三场景等权|78.61%|63.33%|70.28%|37.00%|46.68%|20.00%|15.28pp|2.727|

与同package的v11原head比较，注册前old提升1.67pp、注册后old提升3.89pp、seen-new提升1.50pp、H提升2.23pp、遗忘降低2.22pp；但注册后old仍比direct低6.94pp，远未达到正式门槛，Round1不得晋升。

逐样本机制审计显示：旧类360行中救回25行、损害11行，old→new从108降至88；新类1200行中救回69行、损害51行，wrong-new从605降至531，但new→old从169升至225。说明EvidenceNorm确实压低了新原型hub，却对部分新类过度惩罚。最差新类`4-10`下降23.33pp，旧类`14-10`下降6.67pp；下一轮需要表征边界学习，不能继续单纯放大类尺度校准。

资源闭环：0参数、0epoch、0optimizer step、0额外backbone forward；26类EvidenceNorm部署增量104B，6类before评估比较器24B，formal双流128B。计入完整prototype/transform/bias后，部署head为14,820B、before比较head为3,180B、formal双流为18,000B；FP32实时数组分别29,640B、6,360B、36,000B。加原adapter后星上部署持久状态124,638B，formal双流诊断状态127,818B，均低于256KB。量化后最大逆尺度为9.99634，不超过锁定上限10。

最终预测NPZ SHA=`1ab79ffcb279e9580d48c72f34d04f79f5e7f28987bc7a6140a1cb045b7f325c`，manifest SHA=`a035633664d21b0cc0128583acb826801446637e6f5f66c8041a1b53696c3d06`，truth sidecar SHA=`5a70620a6b90a86ca47b8be1bad83c5e881826d976cd3885b47d0fe6ffde8470`。聚焦测试30项、更广runtime/closure/diagnostic测试61项以及sealed v1 pipeline E2E 1项均通过；独立复审未发现P0/P1。
