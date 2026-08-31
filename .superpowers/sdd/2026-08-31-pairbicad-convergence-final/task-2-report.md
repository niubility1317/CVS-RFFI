# Task 2：PairBiCAD source-LORO收敛评估与早停

## Traceability

|ID|brief要求|目标文件|状态|验证|
|---|---|---|---|---|
|T2-CLI|增加5个`bicad_*`CLI参数及默认值|`code/SSDG/train_ssdg.py`|verified|完整回归|
|T2-BUDGET|`bicad_optimizer_updates=0`保持candidate config；正值校验500倍数且范围4000–9000，并用`dataclasses.replace`覆盖本次预算|`code/SSDG/train_ssdg.py`|verified|预算与9000覆盖测试|
|T2-VALIDATE|启用LORO时校验interval、min_updates、patience和heldout receiver|`code/SSDG/train_ssdg.py`|verified|非法参数及receiver测试|
|T2-PAYLOAD|data context保留同一个`wisig_payload`引用|`code/SSDG/train_ssdg.py`|verified|payload identity测试|
|T2-LORO|用同一payload、训练day和唯一heldout source receiver构建source-only loader，不构建target loader|`code/SSDG/train_ssdg.py`|verified|loader构造测试|
|T2-EVAL|按固定clean/三种LEO顺序、row seed偏移评估accuracy、per-class accuracy和floor|`code/SSDG/train_ssdg.py`|verified|scenario seed/class floor测试|
|T2-SCORE|实现主分数、严格`1e-12`改善和patience状态机|`code/SSDG/train_ssdg.py`|verified|score/patience测试|
|T2-CLOCK|optimizer step后在min_updates起按interval评估，连续patience次无改善真实停止|`code/SSDG/train_ssdg.py`|verified|eval clock测试|
|T2-ARTIFACTS|写不可覆盖评估checkpoint、UTF-8 LF curve和selection JSON；final checkpoint/runtime记录实际及计划预算|`code/SSDG/train_ssdg.py`|verified|artifact/runtime helper测试及编译检查|
|T2-TESTS|覆盖默认/9000、非法预算、receiver校验、clock、score、patience、final row_key和禁止访问flag|`code/tests/phase1_bicad_xr/test_ssdg_entry.py`；`code/tests/phase1_bicad_xr/test_trainer.py`|verified|聚焦12/12，完整回归全部通过|

## Scope boundary

本任务只修改上述owned文件和本报告；不访问target、Phase2、support、query或truth，不启动N607或正式训练，不修改已有plan、报告、分析脚本或其他Agent的工作区改动。

## Implementation result

- P0–P4支持本次运行4000–9000的500倍数预算；`0`保持candidate默认值。
- source-LORO只复用已加载payload，使用source day1/2/3和唯一held-out receiver；评估顺序固定为clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 评估结果记录accuracy、per-class accuracy、floor及固定seed偏移；严格超过`1e-12`才重置patience，连续patience次无改善后停止。
- 早停时最终checkpoint记录实际stop update；`planned_optimizer_updates`和`planned_total_updates`记录planned上限，`candidate_config.optimizer_updates`保留planned上限以支持严格trainer重建；最佳LORO点不替换final checkpoint。
- source-LORO checkpoint使用`xb`不可覆盖写入，curve追加UTF-8 LF，selection JSON包含全部访问flag且均为`false`。

## Verification

工作目录：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`

- Python环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`
- `python -m pytest code/tests/phase1_bicad_xr/test_ssdg_entry.py code/tests/phase1_bicad_xr/test_trainer.py -q -p no:cacheprovider`：exit 0，全部测试通过。
- 聚焦PairBiCAD回归：12/12通过。
- `python -m py_compile code/SSDG/train_ssdg.py`：exit 0。
- 未执行N607、真实数据训练或正式实验；因此未声称真实checkpoint artifact闭合。
