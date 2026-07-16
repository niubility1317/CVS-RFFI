# qKNNV42 K1 support信赖域快速适应报告

## 运行信息

|字段|内容|
|---|---|
|实验ID|`qknnv42_k1_support_trust_adapt_20260716`|
|时间|2026-07-16|
|操作者|Codex主线程|
|状态|`LOCAL_IMPLEMENTATION_IN_PROGRESS`；尚未启动N607|
|目标|用统一`JG-R8`候选消除K1 target梯度相对P4 identity的负迁移，并保持极轻型适应|
|比较对象|strict direct ADV3B02、ADV3B02＋P4 identity、v23 `JP8_LR005`|

## 已知证据

|候选|K/场景边界|accuracy/floor|相对P4 identity|资源|解释|
|---|---|---:|---:|---:|---|
|`JG_R8_LR020`|source receiver、K10、6类、3个LEO_weak View|88.8354%/75.9140%|+0.5133/+2.1505pp|6,400参数、5epoch/50step、1.293s|K10 source winner；不是新类注册结果|
|`JP8_LR005`|同一source split、K1|88.0013%/73.5484%|-0.0321/0.0000pp|3,840参数、5epoch/5step、0.626s|相对direct为正主要来自P4和support prototype；梯度本身为负|

## 假设

K1只有每类一个物理support，3个信道View高度相关。继续提高学习率、扩大层数或增加epoch只会提高方差。为避免按K更换算法，保留K10已胜出的6,400参数`JG-R8`作为统一候选；冻结backbone并缓存`feat_id/feat_pa/frozen feat_joint`，只重算门控和最终投影小子图。训练delta仅作为方向建议，再用未参与梯度的support增强和最差`View×class`margin约束选择缩放系数；证据不足时退回identity。

## 固定配置

|项目|值|
|---|---|
|基座|ADV3B02＋ground P4|
|可写层|`id_backbone.cls_head.id_gate.0＋joint_proj.0`|
|LoRA|rank8、alpha8、6,400参数|
|优化器|SGD、momentum0、weight decay1e-4、grad clip1|
|训练预算|5epoch、K1共5step|
|训练View|3个已注册`leo_*_weak`View|
|缩放网格|`0,0.125,0.25,0.5,0.75,1`|
|禁止项|query训练/选模/阈值、role Oracle、类别配额、dense query图、clean访问|

## 输入、处理、输出

输入是密封K1 support、固定ADV3B02 checkpoint、固定P4和预注册增强配置。处理包括冻结backbone特征缓存、`JG-R8`小子图训练、support-only增强验证、逐View×类margin信赖域和FP16 delta缩放。输出包括原始/缩放delta、选择的`alpha`、loss/margin trace、merge parity、资源回执，以及后续四段式注册链使用的head状态。

## 成功标准

- 本地机制：类置换等变、无query参数、精确6,400个可写参数、FP16 roundtrip/merge parity、`alpha=0`精确恢复P4、资源可重算。
- 开发性能：K1相对P4 identity不再为负，并提高最差类；相对strict direct ADV3B02继续扩大正差。
- 最终性能：按项目目标在5receiver×至少5seed的独立确认中验证paired增益及CI；未完成前只记诊断结果。

## N607计划

只有本地代码、定向测试、Git提交、report更新和direct preflight全部通过后才允许同步。预定远端根、命令、PID、GPU和日志将在实际launcher锁定后填写；当前没有远端状态变更。

## 本地实现进展

|文件|作用|验证|
|---|---|---|
|`paper_reproduction/cvs_aligned/k1_support_trust.py`|留一View×类margin、逐alpha安全审计、最大安全缩放、LoRA residual缩放|定向pytest通过|
|`tests/test_k1_support_trust.py`|无query接口、类置换等变、安全非零选择、回退0、只缩放`lora_b`|7/7通过|
|`cached_jg_real_parity.json`|真实ADV3B02＋P4上完整`feat_joint`与缓存JG小子图等价性|最大绝对误差5.0664e-7≤1e-6，PASS|

实际命令在`ssr-gpu`环境中执行：`python -m pytest tests/test_k1_support_trust.py -q`以及`python -m py_compile paper_reproduction/cvs_aligned/k1_support_trust.py`。第一次无Profile PowerShell未加载Conda hook，属于命令包装失败；加载`conda shell.powershell hook`后验证通过。当前模块尚未接入真实JG-R8 enrollment，也未产生target准确率。

随后在CPU上严格加载本地ADV3B02 checkpoint和P4，注入零残差`JG-R8`，对seed713101生成的`[7,2,256]`探针分别执行完整`z_id/feat_joint`前向与缓存`feat_cls/feat_dac/feat_pa`后仅重算`id_gate＋joint_proj`。batch size为3，共3次full-backbone调用；最大绝对误差为5.066394805908203e-7，通过1e-6门槛。机器可读证据见`cached_jg_real_parity.json`；正式运行仍以N607 enrollment receipt为准。

## 风险

- support增强仍来自同一物理shot，只能构成更强的安全代理，不能替代独立query验证。
- 过严信赖域可能频繁选择`alpha=0`，只能避免负收益，不能保证达到+2pp。
- 注册后旧新类竞争可能改变最优几何，因此必须与新类注册agent的同run before/after结果联合验收。
