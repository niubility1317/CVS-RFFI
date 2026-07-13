# Phase1联合P0与leo_weak测试协议落地追踪

|ID|来源要求|具体要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P0-01|用户协议|后续默认测试增强统一为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`|根目录`项目.md`、`code/post_stage_cli.py`、`code/train.py`|verified|默认值测试及federated integration通过|legacy仅显式diagnostic|
|P0-02|source-episode根因|让TX×rx_day平衡采样真正进入Phase1训练loader|`code/SSDG/train_ssdg.py`、sampler测试|verified|6TX×6domain×3样本=108完整batch测试通过|原flag仅审计未接线，现已接通|
|P0-03|跨域核心|修复prototype memory中domain-align项无模型梯度的问题|`code/cvsrffi/losses.py`、loss测试|verified|当前batch domain center反向梯度非零|同时修复memory-only inter push|
|P0-04|初期open优化|source/direct/proxy/open-gradient从E1启用，base权重显著高于DualGuard16|新launcher与测试|verified|8条命令均从E1启动并通过trainer parser|短warmup4/8epoch|
|P0-05|联合保护|open有效梯度使用双侧预算，防止open不足或压垮closed/DG|新launcher与现有controller测试|verified|controller上下限回归通过|探索0.10-0.28区间，closed scale下限0.90-0.95|
|P0-06|星地与open联合|所有候选同时启用concat_sa、sat CE/KD、TX条件channel pair、source/local/direct/proxy和U_s三态|新launcher与测试|verified|8条唯一命令与共同机制断言通过|不做单机制孤立优化|
|P0-07|实验验证|8张GPU各1个实验，final-only，Phase1 source-only，保存完整指标与endpoint证据|launcher、report、N607|verified|r2的8/8候选连续完成E1-E80并写出终态/heldout；全部由tail状态机提前终止|结果为`NON_PROMOTABLE_DIAGNOSTIC`，不是120epoch完整候选|
|P0-08|tail闭环|固定source-val metric reference不得受绝对目标或U_s/source guard锁死；绝对不安全只阻断promotion/export，相对扩张才驱动训练期状态|`code/cvsrffi/phase1_v2_control.py`、`code/SSDG/train_ssdg.py`、测试|verified-local|tail专项12 passed；新launcher显式禁用tail训练早停并允许绝对未达标metric reference|待N607确认E120和非空p99 delta|
|P0-09|U_s有效路由|`u_geometry_all_valid_queries`必须让trusted core直接进入U direct，all-valid进入U invariance，不再重新与高置信CE mask相交|`code/SSDG/train_ssdg.py`、U_s测试|verified-local|mask语义单测和Phase1回归通过|待N607验证U direct≥80%、U invariance≥95%活跃率|
|P0-10|endpoint对齐|virtual negative固定、metric gate可微，避免通过移动动态virtual构造器伪降DM指标|`code/cvsrffi/losses.py`、direct metric测试、launcher|verified-local|12项direct metric测试通过，detach方向与multiview遥测已覆盖|新launcher固定virtual、开放gate梯度|
|P0-11|分目标梯度预算|在总open/closed预算内，为source episode、invariant core、endpoint boundary、U_s geometry分别分配有效梯度份额|`code/SSDG/train_ssdg.py`、launcher、梯度测试|verified-local|四组raw norm/scale/effective norm单测与命令parser通过|待N607验证各组活跃和不长期卡scale上限|
|P0-12|星地receiver证据|final-only heldout必须在`leo_weak`下导出逐receiver结果、sat receiver floor和明确eval seed|`code/cvsrffi/eval.py`、`code/cvsrffi/ssdg_guard.py`、launcher、测试|verified-local|sat eval/guard 9 passed；新launcher使用`eval_sat_on=all`|待N607检查terminal artifact字段|
|P0-13|调度器终态|父scheduler必须汇总子进程COMPLETE/STOPPED/FAILED并在存在非COMPLETE时返回非零|DualGuard scheduler、测试|verified-local|scheduler状态和return-code专项6 passed|待远端终态验证|
|P0-14|DG冲突保护|open梯度与CE/KD/sat冲突时保护closed方向，只投影open冲突分量|`code/SSDG/train_ssdg.py`、梯度测试、新launcher|verified-local|closed优先级、共享参数scope和梯度非有限bundle测试通过|新矩阵closed scale下限0.95-0.98|

## 反向审计边界

- `leo_weak`测试只证明同简化LEO增强族的独立随机压力鲁棒性，不证明跨实现或真实卫星部署。
- Phase1只评价闭集DG、known几何、proxy风险、U_s路由和prototype/endpoint质量，不声明真实unknown FAR/FPR95。
- 动态DM仍不是最终拒识边界；promotion必须绑定fixed source-val与`endpoint_accept_v1`。

## r2终局反向审计

- 8个候选计划120epoch，实际均在E80触发`tail_safety_fail_closed`；E10-E40积累reference window，E50仅p99略超82度，E60/E70/E80依次进入WARNING/ROLLBACK/STOP。
- 8份fixed source-val终局p95为55.39-57.16度、p99为82.38-84.06度，但`tail_reference_epoch=-1`，所以`p99_delta`从未真正可计算。
- J5同批泛化最好：overall 89.87、strict UDU 86.01、clean receiver floor 75.03、`leo_weak`场景aggregate floor 76.46、sat strict floor 70.49；尚未达到sat strict UDU 78目标。
- `source_episode_overflow`仍为0.97量级；legacy proxy约0.64、bridge=1.0未改善。动态DM低值不能替代endpoint/legacy证据。
- U_s三态覆盖112条query，但trusted core只有约6-7条/batch，随后再次与高置信CE mask相交，导致U direct与U invariance运行时为0。
- 当前heldout artifact没有星地逐receiver结果，因此不能声称sat receiver floor达到目标。

## 启动交接

- Git实现提交：`adaeae9`；scheduler递归修复提交：`1a665db`。
- 首次run未启动训练，递归错误证据保留；正式run为`phase1_dgleo_jointp0_leoweak8r2_20260713`。
- scheduler PID`3706157`；训练PID依次为`3706192/3706281/3706793/3707256/3707719/3708182/3708644/3709106`。
- E1 source-val TX为98.58%-98.61%，8/8无fatal；step rate均0.937且gradient norm聚合为NaN，列为E2-E5必查风险。
