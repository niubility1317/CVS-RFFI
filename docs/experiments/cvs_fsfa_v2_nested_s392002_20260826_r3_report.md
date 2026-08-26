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

待发布、truth-last评分和独立readback后填写。
