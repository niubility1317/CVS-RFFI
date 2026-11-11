# Tri SSL source-calibrated coherence and equal-dose U controls

## Fixed experiment

On the earlier source-U training logs, mean repeat coherence was approximately
0.9972 and `0.25 + 0.75ρ` scaled reference features by about 0.9979. To make
the physical perturbation measurable, this fixed protocol uses the source-scale
mapping `1 - clamp(100*(1-ρ), 0, 0.75)`; this factor is an exploratory choice
based on source-U range, not a calibrated SNR or a channel estimate.

Five prespecified arms each use seeds 2026092701–03 and same-setting original
Tri SSL scratch E20 continued E10+E20: unattenuated extra source-U pseudo CE
with weights 0.25 and 1; source-coherence reference scaling at weights 0.25 and
1; and a weight-1 uniform reference scale matching the adaptive batch mean.
All use the same accepted-U selection, existing fusion head and U gradient
guard, with no new classifier. The unattenuated controls isolate the effect of
the extra CE; the matched uniform arm probes per-packet coherence dependence.
All five models retain time, frequency, PA and reference paths at inference.

This single-source-RX setting was chosen from exposed historical test scores;
earlier candidate results were also observed. The resulting evidence is
exploratory, not a clean blind test. This matrix, source calibration and final
checkpoints are fixed before its predictions; no score-driven changes within it.
After all 15 prediction sets for clean and six practical views are frozen,
score independently and compare every scene with same-packet same-seed Tri and
ordinary PL DURA-LU; report corrected/broken packets, TX/RX/day, accepted U and
actual scaling distributions, runtime and failures. PL's natural temporal order
and training mechanism differ from Tri. User target is >2 percentage points
over PL in each scene plus stronger Tri results; it is not yet achieved.

Sharing up to four N607 experiment processes per GPU was explicitly authorized;
free-memory and process admission remains, and no other owner's process or
outputs may be altered. The per-packet reference-view design differs from the
other chat's source-day pairing and branch-gradient protection. The masked and
unmasked U comparisons follow established semi-supervised regularization ideas;
no innovation or general performance claim is made without supporting evidence.

- run_id：`20261111-phase1-pamf-tri-u-coherence-cal-m1-m15-r01`
- group_id：`phase1-trisig-coherence-calibration-m1`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Five fixed three-seed arms: unattenuated U pseudo-CE at two weights, source-coherence-expanded reference attenuation at two weights, and batch-mean-matched attenuation. Source U coherence near .997 motivated calibration; test data already exposed.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
