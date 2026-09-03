# ADV3B02-DAOT-STN-RX-V2设计落地追踪

## 1.交付状态

- 方法ID：`ADV3B02-DAOT-STN-RX-V2`。
- 当前状态：`LOCAL_VERIFIED/RELEASE_AUTHORIZED`；用户追加授权扩展确认，发布矩阵为`V2-P1～V2-P5`。
- 工作树：`.worktrees/adv3b02-daot-stn-v1`，分支：`codex/adv3b02-daot-stn-v1-20260901`。
- 实现、测试和真实checkpoint无query smoke已完成；用户随后授权发布实验，运行证据以独立实验报告为准。
- 部署默认教师为`clean+rotating channel/receiver fresh+Temporal Orbit Memory`，即每步两次新鲜教师前向；三次新鲜教师前向仅保留给A2/A3上界实验。
- 用户明确排除“使用上一轮checkpoint执行非LEO_WEAK测试”，本实现未提供、未调用这一路径。

## 2.固定数据与评估边界

- 数据集：`Dataset_WigSig/ManySig.pkl`，equalized=`true`。
- split：`tx_rx_day_1_7_2`，seed=`392005`。
- source receiver：`[1,3,4,6,8]`；source day：`[1,2,3]`。
- source pool：90000；`L_s=6300`、`U_s=56700`、单一`V=27000`。不再拆分`V_cal/V_select`。
- 最终评估白名单仅为`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 非LEO_WEAK跨族测试状态：`REJECTED_BY_USER_SCOPE`。

## 3.设计—代码—测试追踪

|ID|报告机制|落地点|运行可达性|验证状态|
|---|---|---|---|---|
|RXV2-01|coverage-aware、clean锚定球面轨道教师|`orbit_teacher.py`、`daot_training.py`|V2目标函数默认启用|单元与入口测试通过|
|RXV2-02|扩展物理可恢复性与缺失mask|`deployment_orbit.py`|教师权重与U_s可信度共同使用|SNR/elevation/K-factor/deep-fade/clip/occupancy/canonical/phase/spectral覆盖|
|RXV2-03|统一方向注册表、稳定弦长敏感度、delta标定|`deployment_orbit.py`、`selective_tangent.py`、`tangent_calibration.py`|V2每批轮换一个pure nuisance方向；clipping/quantization仅secant|单元与入口测试通过|
|RXV2-04|随机单TX干预与Directional Jacobian Routing|`deployment_orbit.py`、`selective_tangent.py`、`train_ssdg.py`|V2在route阶段启用|可重放性、单方向性和路由margin测试通过|
|RXV2-05|Source-only Receiver Style Bank|`receiver_style_bank.py`、`train_ssdg.py`|仅用source receiver在线统计；旋转fresh视图使用|target角色拒绝、状态恢复与变换测试通过|
|RXV2-06|同TX/同激励桶跨RX对齐与尾部风险|`receiver_conditioned_alignment.py`、`train_ssdg.py`|`L_RX`和`L_tail`为独立项|跨TX隔离、可微对齐、组资格与权重上限测试通过|
|RXV2-07|U_s连续可信度与三态选择|`daot_unlabeled_trust.py`、`train_ssdg.py`|Core使用soft目标；Ambiguous仅feature；Irrecoverable不参与伪标签目标|无真实标签输入，class×RX×severity配额测试通过|
|RXV2-08|分支专属不变性预算|`branch_invariance.py`、`train_ssdg.py`|按轮换方向路由到`id_feat_imp/id_feat_pa/id_feat_joint`；缺失分支显式退化为joint|预算差异与超预算损失测试通过|
|RXV2-09|S0—S6独立调度与持久冲突投影|`orbit_teacher.py`、`daot_gradient_control.py`、`train_ssdg.py`|orbit/tangent/route/RX/tail独立scale；连续3次冲突后仅投影`id_backbone`；完整optimizer参数仍保留基础梯度|epoch边界、持久冲突、模型外MUSE head梯度和训练反向测试通过|
|RXV2-10|Tensor Temporal Orbit Memory与两fresh默认|`orbit_teacher.py`、`train_ssdg.py`|V2默认启用；含可靠度、scenario/RX bin、epoch单位TTL和checkpoint状态|memory命中/过期、跨epoch重访、入口两fresh+memory测试通过|
|RXV2-11|选择性nuisance子空间|`selective_nuisance_subspace.py`、`train_ssdg.py`|默认`lambda_subspace=0`严格旁路；开启时每5epoch更新并checkpoint|关闭恒等、开启抑制和状态代码验证通过|
|RXV2-12|结构化batch与source-only选择接口|`balanced_tx_rx_sampler.py`、`daot_source_selection.py`、V2配置|V2强制5个source RX的TX×RX平衡labeled batch；提供30/45/15/10分配与5折source RX选择评分|分配、R≥3、5折和成本惩罚测试通过|
|RXV2-XF|旧checkpoint非LEO_WEAK测试|无|不可达|按用户要求排除|

## 4.目标函数与默认参数

V2训练目标按独立scale实现为：

`L=L_base+λ_zL_orbit,z+λ_logitL_soft+λ_protoL_proto+λ_tanL_tangent+λ_routeL_route+λ_RXL_RX+λ_tailL_tail+λ_cleanL_clean-anchor+λ_subL_subspace`

默认权重：`λ_z=0.40`、`λ_logit=0.075`、`λ_proto=0.125`、`λ_tan=0.035`、`λ_route=0.05`、`λ_RX=0.075`、`λ_tail=0.10`、`λ_clean=0.025`、`λ_sub=0`。各项先经EMA尺度归一化，再乘自身阶段scale和权重；不再把nuisance/fingerprint错误绑到统一orbit scale。

V2默认将旧`nuisance`和`fingerprint keep`权重显式置0，避免夹带未预登记目标；显式CLI loss权重拥有最高优先级，可用于P1/P2等同row相邻消融，未显式给出的权重才从V2 profile补齐。

配置固化于`configs/phase1_adv3b02_daot_stn_rx_v2_s392005.json`。配置状态明确为`IMPLEMENTED_NOT_LAUNCHED`，且`launch.authorized=false`。

## 5.本地验证证据

- Python编译：V2涉及的训练入口、核心模块和smoke脚本均通过。
- V2聚焦测试：101项通过。
- 受V2反向入口影响的既有MUSE/CRRA/legacy测试：66项通过。
- 独立P0/P1审查：无P0，首次发现4个P1；修复后唯一一次定点复审确认4项全部`RESOLVED`，控制测试18/18通过。修复项为跨epoch memory TTL单位、MUSE外部head梯度、显式CLI消融优先级、旧nuisance/fingerprint默认关闭。
- CLI dry-run：seed392005、200epoch、V2开关解析通过；未构造数据或模型，未启动训练。
- 真实checkpoint无query smoke：
  - checkpoint：`E:\type10-7\local_artifacts\adv3b02_ecrs_smoke\best_joint_safe_ssdg.pth`；epoch194。
  - 输入：4条source-shaped合成IQ；`query_inputs=0`、`target_inputs=0`、`evaluated_scenarios=[]`。
  - 输出：`z_id_shape=[4,160]`、`fresh_teacher_forwards=2`、`temporal_memory_views=1`、`memory_hit_rate=1.0`、`receiver_style_ready=true`。
  - artifact：`E:\type10-7\local_artifacts\adv3b02_daot_stn_rx_v2_20260903\real_checkpoint_no_query_smoke_final2.json`；独立JSON读回为`PASS`。
- 仓库级一次性全量收集未进入测试执行：`ssr-gpu`为Python3.10，既有`tools/codex_automation_fallback.py`直接导入Python3.11内置`tomllib`；同时`tests/`与`code/tests/`存在5组基线同名模块。二者均不在本次差异内。本轮未修改无关测试基础设施，改以V2聚焦集和受影响既有集完成可归责回归。

## 6.实现与实验结论边界

- 已证明：代码可导入、目标函数可反向、V1默认路径回归通过、V2默认两fresh+memory路径可达、真实checkpoint兼容且无query smoke通过。
- 尚未证明：任何训练收敛、clean/LEO_WEAK性能提升、运行时吞吐收益、A2/A3上界差距或晋级结论。
- 本轮无实验数据，因此不得把`LOCAL_VERIFIED`解释为性能验证或默认方法晋级。
