# Phase1联合P0与leo_weak测试协议落地追踪

|ID|来源要求|具体要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P0-01|用户协议|后续默认测试增强统一为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`|根目录`项目.md`、`code/post_stage_cli.py`、`code/train.py`|verified|默认值测试及federated integration通过|legacy仅显式diagnostic|
|P0-02|source-episode根因|让TX×rx_day平衡采样真正进入Phase1训练loader|`code/SSDG/train_ssdg.py`、sampler测试|verified|6TX×6domain×3样本=108完整batch测试通过|原flag仅审计未接线，现已接通|
|P0-03|跨域核心|修复prototype memory中domain-align项无模型梯度的问题|`code/cvsrffi/losses.py`、loss测试|verified|当前batch domain center反向梯度非零|同时修复memory-only inter push|
|P0-04|初期open优化|source/direct/proxy/open-gradient从E1启用，base权重显著高于DualGuard16|新launcher与测试|verified|8条命令均从E1启动并通过trainer parser|短warmup4/8epoch|
|P0-05|联合保护|open有效梯度使用双侧预算，防止open不足或压垮closed/DG|新launcher与现有controller测试|verified|controller上下限回归通过|探索0.10-0.28区间，closed scale下限0.90-0.95|
|P0-06|星地与open联合|所有候选同时启用concat_sa、sat CE/KD、TX条件channel pair、source/local/direct/proxy和U_s三态|新launcher与测试|verified|8条唯一命令与共同机制断言通过|不做单机制孤立优化|
|P0-07|实验验证|8张GPU各1个实验，final-only，Phase1 source-only，保存完整指标与endpoint证据|launcher、report、N607|implemented|r2已在N607按GPU0-7各1个启动；E1 8/8完成|待终局指标后改为verified|

## 反向审计边界

- `leo_weak`测试只证明同简化LEO增强族的独立随机压力鲁棒性，不证明跨实现或真实卫星部署。
- Phase1只评价闭集DG、known几何、proxy风险、U_s路由和prototype/endpoint质量，不声明真实unknown FAR/FPR95。
- 动态DM仍不是最终拒识边界；promotion必须绑定fixed source-val与`endpoint_accept_v1`。

## 启动交接

- Git实现提交：`adaeae9`；scheduler递归修复提交：`1a665db`。
- 首次run未启动训练，递归错误证据保留；正式run为`phase1_dgleo_jointp0_leoweak8r2_20260713`。
- scheduler PID`3706157`；训练PID依次为`3706192/3706281/3706793/3707256/3707719/3708182/3708644/3709106`。
- E1 source-val TX为98.58%-98.61%，8/8无fatal；step rate均0.937且gradient norm聚合为NaN，列为E2-E5必查风险。
