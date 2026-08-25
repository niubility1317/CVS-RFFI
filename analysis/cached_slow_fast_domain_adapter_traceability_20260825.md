# Cached Slow-Fast Domain Adapter需求追踪

配套设计：`docs/CVS_CACHED_SLOW_FAST_DOMAIN_ADAPTER_V1_DESIGN_20260825.md`

|ID|来源|验收要求|目标文件|状态|验证|
|---|---|---|---|---|---|
|SF-01|附件§3、§12|冻结ADV3B02并生成source-only特征缓存，缓存不进入部署bundle|`slow_fast_cache.py`、`slow_fast_phase15_entry.py`|verified|角色、物理ID、四view和bundle负测|
|SF-02|附件§4.1、§4.2、§5～6|按类中心化学习域基，支持同物理clean/LEO pair和K分布episodic快更新|`slow_fast_phase15.py`|verified|合成域恢复、两类FAST训练和episode计数测试|
|SF-03|附件§9 V0|实现`COMMON_SHIFT_R4`闭式ridge校正|`slow_fast_adapter.py`、`slow_fast_selection.py`|verified|手算fixture、LOO改善与退化回退|
|SF-04|附件§1、§9 V1|实现`FAST_FILM_R8`，Phase2只更新16个快参数|`slow_fast_adapter.py`|verified|参数schema、训练与梯度路径测试|
|SF-05|附件§9|实现`FAST_LOWRANK_R8`，Phase2只更新24个快参数|`slow_fast_adapter.py`|verified|方向gate schema与训练测试|
|SF-06|附件§4.3、§4.4|floor权重0.2和区间trust约束|`slow_fast_objectives.py`|verified|floor/trust手算边界测试|
|SF-07|附件§7.4|K≥2执行support LOO和`lambda={0,0.25,0.5,0.75,1}`选择，失败回退DA0|`slow_fast_selection.py`|verified|LOO改善、完美基线回退和位移审计|
|SF-08|项目K-shot边界|K=1不制造第二shot，固定DA0回退|`slow_fast_selection.py`|verified|K1零更新测试|
|SF-09|项目Phase2权限|只读取合法support、冻结bundle/原型；query只读且逐样本|`stage2_slow_fast_runner.py`|verified|allowlist、query逐条和状态计数测试|
|SF-10|四状态规范|同row输出`DA0_REG0/DA1_REG0`，REG0新类指标N/A|runner/既有truth-last scorer|verified|prediction schema和现有scorer兼容测试|
|SF-11|附件§12|固定三个候选×三个scene的9-row诊断矩阵|config/matrix|implemented|笛卡尔积、同输入与不可覆盖测试；真实路径待N607只读回读后冻结|
|SF-12|晋级门槛|mean≥+1.0pp、floor≥+0.5pp、单类退化≤5pp|scorer/report|pending|边界值测试|
|SF-13|不合理项修正|实际特征维度由原型宽度决定，不硬编码256|bundle/adapter|implemented|合成宽度和错误宽度负测通过；160维真实smoke待发布后闭合|
|SF-14|排除项|不实现hypernetwork、类条件Adapter、新类support域更新或D92式头|全实现|verified|Phase2配置和bundle严格字段白名单|

当前统计：verified=11，implemented=2，pending=1，deferred=0，rejected=0，blocked=0。24项本方法聚焦测试和18项邻近truth-last scorer回归通过，共42项；唯一未闭合科学问题是Phase1.5域方向能否在固定target query上形成决策级改善，该问题只由9-row truth-last实验回答。
