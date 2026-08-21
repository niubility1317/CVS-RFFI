# Phase1 Core90拼接式LEO_WEAK默认策略封存报告

## 1.结论

自本次提交起，Phase1新建训练默认严格复用`ADV3B02_CORE90_SOFT_E200`的有效星地信道增强配置：

- 训练方式：`clean+satellite`拼接，卫星视图采用TX交叉熵监督，`concat_sat_ce_only=true`。
- 星地监督：有效卫星CE权重为`0.68`、`lambda_sat_cons=0`，从E80开始计入总损失；不默认增加普通星地一致性损失。
- 训练场景：仅默认使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 视图日程：E1–40为`leo_clear_weak,p=0.30`；E41–90为`leo_low_elev_weak,leo_rain_weak,p=0.60`；E91–200为三场景并集`p=0.80`。
- 测试：完成训练后必须独立保留clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四项结果。
- 其他路径：`mixed_orbit`及其他星地场景不再是默认值，仅允许通过显式参数用于历史复现、对照或诊断。
- 历史边界：不追溯改写既有命令、checkpoint、日志或已启动实验。

本次只封存代码和协议默认值，没有启动N607实验，也没有生成新的性能结论。

## 2.Core90事实依据

历史`ADV3B02_CORE90_SOFT_E200`的`checkpoint_args`已核对：

|字段|历史值|本次默认|
|---|---:|---:|
|`use_concat_sat_channel_aug`|`true`|`true`|
|`concat_sat_ce_only`|`true`|`true`|
|`concat_sat_ce_weight`|`1.0`|SSDG为`1.0`；通用centralized入口按其直接乘权语义设为`0.68`|
|`lambda_sat_cls`|`0.68`|`0.68`|
|`lambda_sat_cons`|`0.0`|`0.0`|
|`sat_cons_start_epoch`|`80`|`80`|
|`eval_sat_channel`|`true`|`true`|
|`eval_sat_scenarios`|三种`LEO_WEAK`|三种`LEO_WEAK`|

历史解析依据：`E:/type10-7/automation_reports/CV-SincNet/adv3b02_direct_old_strict_20260714_181100/artifacts/resolved_config.json`。

## 3.需求追踪

|需求ID|用户要求|实现位置|验证|
|---|---|---|---|
|P1-SAT-001|Phase1星地增强默认使用Core90拼接路径|`code/training_controls.py`、`code/SSDG/train_ssdg.py`、`code/train.py`|两个训练入口解析默认值测试|
|P1-SAT-002|训练只默认使用`LEO_WEAK`族|统一默认场景与三阶段schedule|默认列表及schedule断言，不含`mixed_orbit`|
|P1-SAT-003|星地测试只默认使用`LEO_WEAK`族|统一`eval_sat_channel=true`及三场景列表|解析测试断言|
|P1-SAT-004|其他星地场景默认不使用|默认策略不包含其他场景；显式诊断覆盖仍保留|显式覆盖测试|
|P1-SAT-005|协议与代码共同封存|`E:/type10-7/项目.md`、`docs/PROJECT_PROTOCOL.md`、Git入口`项目.md`|文本复核、Git提交和远端OID核对|

## 4.变更文件

|文件|用途|
|---|---|
|`code/training_controls.py`|集中定义Core90星地默认策略并应用到已注册CLI字段|
|`code/SSDG/train_ssdg.py`|SSDG Phase1入口应用统一默认策略|
|`code/train.py`|默认centralized Phase1入口应用同一有效权重与E80起点；显式联邦算法路径不在本次Core90默认范围|
|`code/tests/test_phase1_core90_satellite_defaults.py`|验证策略、两个训练入口及显式诊断覆盖|
|`docs/PROJECT_PROTOCOL.md`|公开协议正文固化拼接方法、日程和场景边界|
|`项目.md`|Git协议入口增加Phase1默认摘要|
|`E:/type10-7/项目.md`|运行时source of truth同步相同科学口径|

## 5.验证记录

TDD红灯：

- 新测试首次收集失败：缺少`PHASE1_CORE90_SAT_DEFAULTS`，证明测试能够捕获未实现状态。
- 启用完整默认场景后，旧的单参数`mixed_orbit`诊断覆盖测试失败；已修复解析优先级，避免默认三场景掩盖显式历史诊断参数。
- 独立审查发现通用centralized入口直接使用`concat_sat_ce_weight`，已通过路径适配把有效CE权重固定为`0.68`。
- 独立审查发现默认三场景会掩盖显式单场景`leo_clear_weak`，已改为运行期默认解析完整`LEO_WEAK`，任意显式单场景均可覆盖。
- 新增E80统一测试首次因缺少监督起点常量失败，现已让SSDG与默认centralized入口均从E80启用卫星监督。

当前已通过：

- Python编译检查：`code/training_controls.py`、`code/SSDG/train_ssdg.py`、`code/train.py`全部通过。
- 聚焦回归：67项通过，覆盖默认策略、Core90拼接、CRRA、NTRS、共享评测参数和显式诊断覆盖。
- 独立P0/P1审查：初审4项P1中2项修复；E80由历史Core90权威报告确认保留；联邦为显式不同算法路径，不属于本次默认。定点复审结论为P0=0、剩余P1=0。
- Git提交、push与远端OID核对将在本报告末尾追加。

## 6.封存状态

- 当前状态：`LOCAL_VERIFIED_PENDING_GIT_SEAL`
- N607：未同步、未启动、未改动。
- 性能结果：无新增实验，不作性能提升声明。
