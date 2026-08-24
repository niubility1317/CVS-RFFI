# CVS_META_ADAPTER_TRI_R4_V1 P1 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_p1_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 生产代码提交：`cef0ee30a998a3f2acfcf52c257edb0d19f1e575`

## 候选与矩阵

- P4 Meta-SGD、P3固定LR和P2 source监督adapter均完成Target5，聚合旧类均值与floor变化均为0.00pp。P1是本组最后一个Phase1完成候选，使用随机adapter初始化bundle。
- 仅替换P1 bundle和冻结原型；固定Target5 receiver`20-1`、seed`392002`、五个operating point、三类LEO weak，共15个同row。
- 继续复用同一`p2_min_v1`、`VALIDATED_ONCE`received IQ、物理样本、support标签、capsule、split和query IDs，不重验数据。
- Phase2不读取source／clean样本、source cache、query真值或query角色；query只读；原编码器adapter真实反向传播3步，训练参数≤1%，冻结原型余弦判决规则不变；无D92／LDA／协方差／持久分类头。

## 输入、输出与规则

- P1 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P1/selected_meta_bundle.pt`
- P1冻结原型：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P1/frozen_prototypes.npz`
- 工厂计划：`configs/stage2_meta_adapter_target5_p1_s392002_20260825_r1.json`
- 复用已核对release：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2_scorerfix1/checkout`；归档SHA256=`467aef9c963b3842f3e5ccf89258fa8c4d0d198dd85f66b4a5e8fb687d02fc78`。
- 工厂root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_p1_s392002_20260825_r1`
- smoke root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p1_s392002_20260825_r1_smoke`
- prediction root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p1_s392002_20260825_r1`
- stdout：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_p1_s392002_20260825_r1.out`
- 先执行真实checkpoint无query smoke，prediction完整后才连接truth。旧类均值≥+1.0pp且floor≥+0.5pp才晋级Target25，否则`SCIENTIFIC_FAILURE_NO_PROMOTION`，并结束P1～P4同机制候选组。

## 实际闭合与结果

- P1 plan本地／远端SHA256均为`35bab47b629a9abd83e88ceb5ec888dba14398fc051b804e257743dafb264078`。工厂15-row truth-free receipt与真实checkpoint无query smoke通过：严格加载、3次反向传播、0.8192%训练参数、query／source均未打开。
- 唯一一次15-row prediction矩阵自然完成，矩阵receipt为`PREDICTIONS_COMPLETE`、`truth_opened=false`、`source_opened=false`；独立scorer在prediction闭合后生成15个`score.json`和`target5_summary.json`。
- 三类场景在五个operating point上的DA0_REG0→DA1_REG0结果分别为：`leo_clear_weak`旧类均值68.3333%→68.3333%、floor30.00%→30.00%；`leo_low_elev_weak`为62.50%→62.50%、35.00%→35.00%；`leo_rain_weak`为67.50%→67.50%、45.00%→45.00%。
- 每row均执行3次真实反向传播，但score最大绝对变化仅约0.000002414，15／15 row最终类别均未变化。聚合`mean_delta_pp=0.0`、`floor_delta_pp=0.0`，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`。
- P1不进入Target25。P1～P4同机制候选组全部完成且均未达到门槛：问题不是缺少训练，而是“同时更新time／freq／fusion adapter+当前support目标+冻结原型判决”无法产生稳定的旧类决策收益。下一候选必须改变少层选择或support目标，不再重复交换同类bundle。
