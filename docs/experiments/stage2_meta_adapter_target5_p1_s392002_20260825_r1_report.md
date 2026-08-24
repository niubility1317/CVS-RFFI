# CVS_META_ADAPTER_TRI_R4_V1 P1 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_p1_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
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
