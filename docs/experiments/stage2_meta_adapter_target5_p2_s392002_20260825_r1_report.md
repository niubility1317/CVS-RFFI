# CVS_META_ADAPTER_TRI_R4_V1 P2 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_p2_s392002_20260825_r1`
- 状态：`ANALYZED / SCIENTIFIC_FAILURE_NO_PROMOTION`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 生产代码提交：`cef0ee30a998a3f2acfcf52c257edb0d19f1e575`

## 候选与固定矩阵

- P4 Meta-SGD与P3固定LR均完成3步真实更新并改变score，但Target5的15／15 row最终类别均零变化，聚合均值和floor均为0.00pp。
- 下一候选P2使用Phase1 source监督adapter bundle；仍只更新原编码器内time、freq和fusion adapter，不新增或训练D92式协方差、LDA或持久分类头。
- 固定Target5 receiver`20-1`、seed`392002`、五个operating point和三类LEO weak，共15个同row；仅替换P2 bundle和冻结原型。
- 数据继续使用同一`p2_min_v1`、`VALIDATED_ONCE`received IQ、物理样本、support标签、capsule、split和query IDs，不因候选变化重验。
- Phase2不读取source／clean样本、source cache、query真值或query角色；query只读；正式3步，训练参数≤1%，冻结原型余弦判决规则不变。

## 输入、输出与执行

- P2 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P2/selected_meta_bundle.pt`
- P2冻结原型：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P2/frozen_prototypes.npz`
- 工厂计划：`configs/stage2_meta_adapter_target5_p2_s392002_20260825_r1.json`
- 复用已核对release：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2_scorerfix1/checkout`；远端归档SHA256=`467aef9c963b3842f3e5ccf89258fa8c4d0d198dd85f66b4a5e8fb687d02fc78`。
- 工厂root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_p2_s392002_20260825_r1`
- smoke root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p2_s392002_20260825_r1_smoke`
- prediction root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p2_s392002_20260825_r1`
- stdout：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_p2_s392002_20260825_r1.out`
- 生产代码已通过70项Stage2直接回归和199项邻近回归。先执行真实checkpoint无query smoke；prediction完整后才连接truth。
- 晋级门槛：`DA1_REG0-DA0_REG0`旧类均值≥+1.0pp且floor≥+0.5pp；否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一候选。技术停止不含低性能。

## 实际闭合与结果

- P2 plan本地／远端SHA256均为`14784f82e8bcc299b6d07cfa5a84b59702b56de0f84aa6465d4fe8f158a98ef0`。工厂15-row truth-free receipt与真实checkpoint无query smoke通过：严格加载、3次反向传播、0.8192%训练参数、query／source均未打开。
- 唯一一次15-row prediction矩阵自然完成，矩阵receipt为`PREDICTIONS_COMPLETE`、`truth_opened=false`、`source_opened=false`；独立scorer在prediction闭合后生成15个`score.json`和`target5_summary.json`。
- 三类场景在五个operating point上的DA0_REG0→DA1_REG0结果分别为：`leo_clear_weak`旧类均值68.3333%→68.3333%、floor25.00%→25.00%；`leo_low_elev_weak`为62.50%→62.50%、25.00%→25.00%；`leo_rain_weak`为69.1667%→69.1667%、45.00%→45.00%。
- 3个row的完整query prediction各变化1条，但truth-last旧类均值和floor仍没有变化；其余12个row类别零变化。聚合`mean_delta_pp=0.0`、`floor_delta_pp=0.0`，结论为`SCIENTIFIC_FAILURE_NO_PROMOTION`。
- P2不进入Target25；下一候选为Phase1 P1随机adapter。若P1仍失败，则P1～P4这一组“全time／freq／fusion adapter、三步更新、冻结原型判决”路线整体结束，下一轮需改变少层选择或support目标，而不是继续交换同类bundle。
