# D105-FTU3 source-held gate有符号整数修复

## 1.问题与证据边界

R7的8400行fixed256 strict tap及reference parity已经通过，prediction、truth-open和score各生成1份；`derive-gate`在输出gate前以`source-held derived gate integer drift`退出，因此R7永久保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。根因复算只检查字段类型和符号，不读取或报告准确率、H、floor等未闭合性能值：

- 7个普通计数字段均为原生非负`int`；
- `receiver_held_min_net_correct`和`class_loco_min_net_correct`均为原生负`int`；
- 原验证器把这两个允许为负的“最小净正确数”与普通非负计数放在同一检查组。

## 2.最小修复

`D105-FTU3`只修改`_validate_derived_source_held_gate`的值域：

- 7个普通计数继续要求`type(value) is int and value>=0`；
- 两个`*_min_net_correct`要求`type(value) is int`，允许负、零、正；
- `bool`、`float`和`np.int64`仍拒绝；
- gate计算、schema、字段、阈值、方法参数、证据顺序和生命周期均未改变。

负`min_net_correct`仍使相应`*_all_noninferior=False`，组件保持`DIAGNOSTIC_STATUS`，authority/formal seal拒绝。因此本修复只消除错误的技术失败，不把真实负向方法证据改写为gate通过。

## 3.验证

|验证面|结果|
|---|---|
|有符号字段正向|两个字段的负原生`int`均可完成derive|
|类型负测|两个字段各自的`bool`、`float`、`np.int64`全部拒绝|
|普通计数负测|7个普通计数字段逐项设为负数均拒绝|
|证据顺序|真实`prediction→truth-open→score→derive`测试链闭合|
|组件与封印|负证据形成`DIAGNOSTIC_STATUS`；精确缺失项保留；formal seal拒绝且无输出|
|定向回归|15/15通过|
|统一回归|10个D105/LPO-RC测试文件共253项，执行到100%，无失败或错误|
|静态验证|修改文件`py_compile`通过；`git diff --check`通过|
|canonical身份|54/54 runtime成员通过loader；runtime/method lock保持单行、无CR/LF|

统一回归唯一警告仍是既有`torch.cuda.amp.autocast`弃用警告。

## 4.独立审查

独立Terra max首轮审查结论为`GO / P0=0 / P1=0 / P2=1`。审查沿`derive→build→persist→load→seal`追踪，未发现第二处对两个有符号字段施加非负限制；P2仅要求补充组件状态和formal seal拒绝的端到端断言。增量独立复审已实跑该新增case，并核对canonical runtime/method lock，最终结论为`GO / P0=0 / P1=0 / P2=0`。

## 5.文件身份

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d105_phase1_bundle.py`|`ae7d16b64ae475f4aa04de78da2283fee7cee97b51a1e3a02a9ada426edc1876`|
|`tests/test_stage2_d105_phase1_bundle.py`|`9f5739493f82064543ad65dc2006c6c58953574c1455ce0d2948a8326079b546`|
|`configs/d105_candidate_runtime_manifest_20260731.json`|`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`|
|`configs/d105_candidate_method_lock_20260731.json`|`7345f81e88588c46ad453eb315786306f28291478a5eaddce618ef7ee6998ecd`|

本地验证不授权N607、authority、formal seal或Target25。下一门是本地Git提交后的精确archive复核；新运行必须使用全新run ID，不能恢复、覆盖或重标R7。

实现提交：`230c6cbc9149250ca0303ca240945d0e0992360e`。
