# TD-HTRC模块二改造实施追踪

日期：2026-08-19

来源：用户提供的《目标域分层传输—稳健中心模块》改造说明。

范围：本次实现M2.1和M2.2两个显式可选模块二升级入口；默认D92 E0路径不改变。M2.2采用低自由度低秩/块尺度传输、对角中心后验不确定性和目标扰动谱收缩；物理nuisance/z-dom融合仍暂不接入。

|ID|来源要求|具体验收项|目标文件|状态|验证|
|---|---|---|---|---|---|
|TDHTRC-01|四层结构/13.1|使用旧类ground-target中心配对估计共享目标域偏移|`code/cvsrffi/stage2_td_htrc_target_transport.py`、`code/scripts/probe_d92_registration_balanced_covariance.py`|verified|`test_shared_offset_uses_old_class_pairs...`、D92集成测试|
|TDHTRC-02|4.1–4.2|目标旧类中心由target support计算，并保留类内Cauchy可靠性与有效样本数|`code/cvsrffi/stage2_td_htrc_target_transport.py`|verified|合成K=1/K=5测试通过|
|TDHTRC-03|4.3、13.1|不把共享偏移强制投影到地面扰动谱，保留谱外分量|`code/cvsrffi/stage2_td_htrc_target_transport.py`|verified|谱外分量审计和断言通过|
|TDHTRC-04|7|support使用共享偏移规范化；通过等价截距编译保证raw query使用同一变换|`code/cvsrffi/stage2_td_htrc_target_transport.py`|verified|raw-query截距等价测试通过|
|TDHTRC-05|8.3、13.1|输出共享偏移不确定性、共享解释率和谱覆盖率审计量|`code/cvsrffi/stage2_td_htrc_target_transport.py`|verified|偏移协方差正定及审计字段测试通过|
|TDHTRC-06|Phase2协议|只读取support、旧类地面聚合中心和不可变扰动谱；不读取query或query truth|`code/cvsrffi/stage2_td_htrc_target_transport.py`、测试|verified|接口无query参数；`query_rows_used=0`|
|TDHTRC-07|13.1|保留现有类内Cauchy稳健中心；K=1/2不因类内方差不足而失败|`code/cvsrffi/stage2_td_htrc_target_transport.py`、测试|verified|K=1/K=2及既有D81回归测试通过|
|TDHTRC-08|实现边界|通过显式opt-in builder接入D92组件，不改变默认D92 E0和既有结果语义|`code/scripts/probe_d92_registration_balanced_covariance.py`、`code/cvsrffi/stage2_ablation_executors.py`、测试|verified|默认/opt-in两组执行器测试通过|
|TDHTRC-09|13.2及后续|物理状态融合、z-dom连续状态和自由全维适配器|本次不修改|deferred|当前Phase2没有冻结的连续域状态映射；自由全维矩阵不由6个锚点辨识|
|TDHTRC-10|5、13.2|M2.2在地面扰动基上拟合正则低秩传输，并在有完整Phase1中心时估计三块尺度|`code/cvsrffi/stage2_td_htrc_m22.py`、`code/scripts/probe_d92_registration_balanced_covariance.py`|verified|M2.2核心测试和D92探针测试通过|
|TDHTRC-11|8|M2.2对旧类使用ground先验、对新类不伪造类别先验，输出后验中心和对角不确定性|`code/cvsrffi/stage2_td_htrc_m22.py`、`code/cvsrffi/stage2_d92_registration_balanced_covariance.py`|verified|后验形状、正值方差和旧/新类先验断言通过|
|TDHTRC-12|9|由旧类跨域残差与地面谱收缩构造目标自适应扰动谱，并用于最终Cauchy权重|`code/cvsrffi/stage2_td_htrc_m22.py`|verified|自适应谱保留秩和最终稳健中心测试通过|
|TDHTRC-13|12|将后验中心不确定性作为对角项加入D92任务共享协方差|`code/cvsrffi/stage2_d92_registration_balanced_covariance.py`|verified|D92探针审计显示不确定性已启用且trace为正|
|TDHTRC-14|实现边界|通过`module2_mode="td_htrc_m22"`显式启用，默认D92 E0保持不变|`code/cvsrffi/stage2_ablation_executors.py`、测试|verified|M2.2执行器测试通过；默认路径回归通过|

## 输入输出约定

- 输入：`rows[N,288]`目标域support特征、`labels[N]`整数注册标签、与旧类注册表对齐的6个旧类160维Phase1聚合中心、可选完整288维旧类中心、160维地面扰动基及谱权重。
- 输出：用于D92组件拟合的传输规范化support、目标自适应谱、后验中心/对角不确定性、已编译到raw-query坐标的affine head和审计指标。
- 查询边界：query只使用最终head进行评分，不参与偏移估计、权重估计或任何状态更新。

## 证据边界

本次只证明代码路径和合成单元性质，不声称目标域适应性能已经提升。任何性能结论必须来自同一row、同一support/query划分的后续paired实验。

## 已运行验证

```text
python -m py_compile code/cvsrffi/stage2_td_htrc_target_transport.py code/cvsrffi/stage2_td_htrc_m22.py code/cvsrffi/stage2_d92_registration_balanced_covariance.py code/cvsrffi/stage2_ablation_executors.py code/cvsrffi/stage2_ablation_factory.py code/scripts/probe_d81_ground_nuisance_cauchy_center.py code/scripts/probe_d92_registration_balanced_covariance.py
python -m pytest -q tests/test_stage2_td_htrc_target_transport.py tests/test_stage2_td_htrc_m22.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d81_ground_nuisance_cauchy_center.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_probe_d81_ground_nuisance_cauchy_center.py tests/test_stage2_ablation_executors.py tests/test_stage2_ablation_factory_catalog.py
```

两次命令均在`ssr-gpu`环境完成；聚焦及回归集合均通过。M2.1/M2.2均为显式opt-in，不代表现有D92 E0的默认screen结果已经替换。
