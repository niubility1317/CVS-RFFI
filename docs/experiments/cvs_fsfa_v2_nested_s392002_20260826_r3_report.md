# CVS-FSFA-V2因子化解析适配实验报告

## 预登记

- run ID：`cvs_fsfa_v2_nested_s392002_20260826_r3`
- 候选：`A0`、`B3(receiver rank4+LEO rank4闭式域码)`、`B5(B3+解析support→query元训练)`
- 矩阵：7个source outer receiver×4个场景×10个独立K10 support draw，共280个feature-level episode；每类query固定10个物理样本。
- 数据边界：仅使用`L_s`地面缓存训练和source nested评价；outer receiver从慢基拟合和inner ridge选择中完全排除。Phase2 smoke只读取既有`p2_min_v1/VALIDATED_ONCE`旧类support，不具备query输入能力，不产生目标性能结论。
- Git实现基线：`aa772dc138e903561398ae6d87835fd009156d1f`；r3仅把不适用的support子组审计从非有限哨兵改为JSON`null`，门控、方法、矩阵和科学变量不变。
- 环境/CWD：N607`/home/szu2070436088/2510044040/CV-SincNet`；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 输入cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt`
- 输入原型/FILM bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt`
- 输出root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_fsfa_v2_nested_s392002_20260826_r3`，不可覆盖。
- GPU：source feature实验使用CPU；真实checkpoint无query smoke使用GPU0，不占用训练槽。
- 预期artifact：`predictions.json`、`score.json`、`CVS_FSFA_V2_B3.int8.pt`、`smoke.json`、`stdout.log`。
- 技术停止规则：输出root碰撞、错误checkout、协议/角色越界、无法生成完整prediction、同一确定性异常在至少两行重复或scorer无法闭合时停止；低性能不停止。
- 科学停止/转向规则：若receiver中位数mean变化≤0、最差receiver低于预登记容差、support/query Spearman<0.2或low-elevation/rain长期低coverage，则最终embedding路线不晋级，转向中间层Adapter候选。
- 晋级阈值：独立目标尚缺失；source结果只决定是否值得等待新capsule。未来目标要求聚合mean≥+1.0pp、floor≥+0.5pp、worst scene mean≥-0.5pp、任一scene/class≥-5pp且新类侵入不恶化。

## 冻结命令

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/evaluate_factored_slow_fast.py --ground-cache /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt --film-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/FAST_FILM_R8.pt --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_fsfa_v2_nested_s392002_20260826_r3/predictions.json --k-shot 10 --draws 10 --query-per-class 10 --seed 392002 --rank-receiver 4 --rank-leo 4 --meta-steps 50 --inner-ridge-grid 0.03 0.1 0.3

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/score_factored_slow_fast.py --predictions /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_fsfa_v2_nested_s392002_20260826_r3/predictions.json --ground-cache /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_cached_slow_fast_phase15_s392002_20260825_r3/ground_feature_cache.pt --output /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_fsfa_v2_nested_s392002_20260826_r3/score.json
```

## 结果

### 1.结论先行

本轮最高交付状态为`ANALYZED`。r3完整产生280个truth-blind source prediction episode，并由独立scorer在prediction闭合后连接truth。最终状态为`SOURCE_NESTED_CALIBRATED_TO_ABSTAIN`，策略选择`A0`：B3和B5均未通过硬可行性门槛，不能晋级，也不进入多seed或目标完整矩阵。

这不是“适配框架不可行”的结论，而是对“在最终embedding上用receiver rank4+LEO rank4的8维解析平移”这一具体路线的否定证据。B3/B5的平均变化接近0，但support收益与query收益相关性为负、伪新类侵入增加，并存在明显单类尾部退化；继续在最终embedding调ridge、步数或门限，优先级低于把Adapter前移到time/frequency/fusion中间层。

目标域性能保持`UNKNOWN_MISSING_INDEPENDENT_TARGET_CAPSULE`。本轮Phase2只用既有合法old-support完成无query smoke，未读取任何目标query；不得把旧rx20-1 query结果复用为本轮独立收益。

### 2.实现与优化落地

- 新增receiver慢基：仅用clean source样本，按类切向残差和robust聚合学习rank4子空间；每个outer receiver完全排除。
- 新增LEO慢基：只用同physical sample的clean/LEO配对残差，先投影去除receiver子空间，再学习rank4场景扰动子空间。
- 分离几何中心与冻结决策原型：几何中心只负责估计域码，决策原型只负责分类，避免角色混用。
- 新增8维闭式域码：每个旧类独立ridge求解，再以几何中位数聚合；Phase2新类support不得参与域码。
- 新增full-support安全缩放：依次尝试1、0.75、0.5、0.25和0；检查正确→错误翻转、正确margin Q10、错误样本margin中位数、class CVaR20、basis coverage和类间域码一致性，失败回退DA0。
- 新增B5解析元训练：source support→query目标包含CE、floor、margin和pseudo-new惩罚；outer receiver query不参与拟合。
- 新增嵌套评价与truth-last scorer：7个outer receiver×4场景×10个独立K10 draw；prediction生成器不读取truth，scorer后连truth并保持同row判定。
- 新增部署bundle和Phase2 runner：慢基/几何中心以int8聚合保存，fast状态只有8个参数，无optimizer state，query更新计数固定为0。
- 新增严格`DA0_ONLY`路径：不打开support、不调用选择器、不应用Adapter，DA0_REG0和DA1_REG0预测逐值一致，全部适配计算账本为0。

### 3.本地验证与独立审查

- `ssr-gpu`聚焦回归：44项全部通过。
- 新增/改动模块`py_compile`：通过。
- `git diff --check`：通过。
- 唯一独立P0/P1审查最初发现pseudo-new prediction生成阶段按query truth筛样本；修复为生成全query侵入分数、独立scorer连truth后筛选，定点复审为`FIXED`。
- N607真实checkpoint无query smoke：`SMOKE_PASS`；`source_opened=false`、`query_input_capability=false`、`query_opened=false`、`query_truth_opened=false`、`query_role_opened=false`，old-support物理样本数60，最终安全选择scale=0。

### 4.发布与运行证据

- 实现提交：`aa772dc138e903561398ae6d87835fd009156d1f`。
- r3修复/冻结提交：`9afae5c778581aa401d7582942790484e8eac010`。
- r3完整release归档：`cvs_fsfa_v2_full_9afae5c7.tar.gz`，35,653,222B。
- 本地/远端唯一SHA256：`b299bafdd2827662806c79eb3600d631964586877bb6bd2ea08c6a10ffcacbea`，一致。
- 远端编译与evaluator入口导入：PASS。
- 正式PID：3560115；CWD为`/home/szu2070436088/2510044040/CV-SincNet`，cmdline、cache、FILM bundle和r3输出root均与预登记一致；CPU运行且未占GPU。
- prediction：35,436,440B，280个episode、7个outer receiver，`query_truth_opened=false`、`target_support_used=false`、`target_query_used=false`。
- scorer：prediction闭合后独立运行，`truth_opened_after_predictions_validated=true`。

### 5.source nested总体结果

|策略|平均Δmean(pp)|平均Δfloor(pp)|receiver LCB90(pp)|最差receiver(pp)|最差episode floor(pp)|最差episode class(pp)|support/query Spearman|最大伪新类侵入Δ|可行性|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|A0|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|PASS|
|B3|-0.0119|+0.0714|-0.0372|-0.0833|-10.0000|-10.0000|-0.0191|+0.3059|FAIL|
|B5|-0.0357|-0.1429|-0.0784|-0.1250|-20.0000|-20.0000|-0.1638|+0.3029|FAIL|

B3虽然平均floor略升，但同一策略的query均值、相关性、伪新类侵入和尾部class/floor不满足门槛，不能抽取单个正指标宣称收益。B5的source元目标从所有outer receiver上均下降，但真实nested分类结果反而更差，说明当前元目标与最终决策效用不一致。

### 6.逐场景结果

|策略|场景|平均Δmean(pp)|平均Δfloor(pp)|最差class(pp)|
|---|---|---:|---:|---:|
|B3|clean|0.0000|0.0000|0.0000|
|B3|leo_clear_weak|-0.0238|-0.1429|-10.0000|
|B3|leo_low_elev_weak|+0.0476|+0.5714|-10.0000|
|B3|leo_rain_weak|-0.0714|-0.1429|-10.0000|
|B5|clean|-0.0238|0.0000|-10.0000|
|B5|leo_clear_weak|-0.1190|-0.4286|-20.0000|
|B5|leo_low_elev_weak|+0.0238|+0.1429|-10.0000|
|B5|leo_rain_weak|-0.0238|-0.2857|-10.0000|

`leo_low_elev_weak`的均值有轻微正变化，但B3/B5在同场景仍存在-10pp单类尾部，且其他场景为负，因此不能晋级。

### 7.逐receiver平均Δmean

|receiver|B3(pp)|B5(pp)|
|---|---:|---:|
|0|0.0000|-0.1250|
|1|0.0000|-0.0417|
|2|+0.0417|+0.1250|
|3|0.0000|-0.1250|
|4|-0.0833|0.0000|
|5|+0.0417|≈0.0000|
|6|-0.0833|-0.0833|

receiver2的B5增益没有跨receiver复现，receiver0/3出现对称负值，进一步支持不推广。

### 8.门控、basis与元目标诊断

- B3选择scale=0的episode为228/280，scale=1为49/280，scale=0.75为3/280；B5分别为188/280、0/280、2/280、15/280、75/280（scale=0、1、0.75、0.5、0.25）。
- B3/B5因`WRONG_MARGIN_MEDIAN`拒绝分别为181和158次；平均basis coverage仅0.0128和0.0358，说明最终embedding的8维子空间对实际support公共残差解释不足。
- LEO共享basis平均解释率：clear 0.4894、low-elevation 0.5017、rain 0.4890。
- 场景主角均值：clear-low 77.21°、clear-rain 77.51°、low-rain 27.03°。clear与另两类扰动接近正交，单一共享rank4 LEO子空间存在明显结构瓶颈。
- 三个ridge(0.03/0.1/0.3)在所有inner LORO中的source gain均为0，最终按确定性顺序选择0.03；当前问题不是ridge取值精调可解决。
- B5在7个outer receiver上都降低了训练元损失，但support/query Spearman为-0.1638，证明优化目标与真实query决策效用错位。

### 9.Phase2与四状态边界

本轮只闭合`DA0_REG0`/`DA1_REG0`的无query运行能力。未执行REG1，因此新类准确率和old/new harmonic均为`N/A`。真实smoke中安全门控选择scale=0，故该old-support capsule上的`DA1_REG0`实际回退为DA0；没有目标query就不能报告目标old-class准确率或DA收益。

### 10.技术失败记录

- r1：增量release遗漏服务器旧代码面缺失的依赖，bundle构建前停止；`NO_PERFORMANCE_RESULT`。
- r2：完整release、smoke和正式计算完成，但严格JSON在持久化“不适用”的`+inf`审计值时停止；partial prediction未评分，`NO_PERFORMANCE_RESULT`。
- r3：以全新不可覆盖root运行，prediction与score完整闭合；r1/r2均未覆盖、删除或复用。

### 11.最终决策与下一步

本轮不推广B3/B5，保持A0。下一候选应落实FSFA-20：把同样的source-only、old-support-only、query只读解析域码注入位置前移到time/frequency/fusion中间层，首轮继续单seed Target5/Target25或更小同row可证伪矩阵。z-dom、CFO/SNR/PSD先验和REG1交互仍保持deferred，不作为下一轮发布gate。取得新的合法目标capsule前，只能继续地面source验证和无query smoke，不能声称目标性能改善。

### 12.正式artifact

- N607 root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_fsfa_v2_nested_s392002_20260826_r3`
- 本地score：`automation_reports/CV-SincNet/cvs_fsfa_v2_nested_s392002_20260826_r3/artifacts/score.json`
- 本地smoke：`automation_reports/CV-SincNet/cvs_fsfa_v2_nested_s392002_20260826_r3/artifacts/smoke.json`
- 本地完整prediction：`automation_reports/CV-SincNet/cvs_fsfa_v2_nested_s392002_20260826_r3/artifacts/predictions.json`
