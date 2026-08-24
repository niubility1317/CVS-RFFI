# CVS_META_ADAPTER_TRI_R4_V1 P3 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_p3_s392002_20260825_r1`
- 状态：`LOCAL_VERIFIED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 生产代码提交：`cef0ee30a998a3f2acfcf52c257edb0d19f1e575`

## 候选与动机

- P4 Meta-SGD在Target5完成3步真实更新并改变score，但15／15 row均未改变最终类别，聚合均值和floor均为0.00pp，因此不晋级。
- 下一候选P3使用Phase1已完成的FOMAML固定LR bundle；其可训练层仍为原编码器内time、freq和fusion adapter，不新增或训练D92式协方差、LDA或持久分类头。
- 假设：P3固定LR比P4的learned LR产生更大但仍有界的support更新，可能跨越冻结原型余弦判决边界。

## 固定矩阵与边界

- Target5 receiver：`20-1`；seed：`392002`；operating point：`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`；三类LEO weak，共15个row。
- 仅替换P3 bundle和冻结原型；复用与P4完全相同的`p2_min_v1`、`VALIDATED_ONCE`received IQ、物理样本、support标签、capsule、split和query IDs，不重新验证数据。
- Phase2不读取source／clean样本、source cache、query真值或query角色；query只读且不更新状态。
- 正式更新3步，训练参数≤1%；判决规则继续使用同一冻结原型余弦规则。

## 固定输入与输出

- P3 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P3/selected_meta_bundle.pt`
- P3冻结原型：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5/P3/frozen_prototypes.npz`
- 工厂计划：`configs/stage2_meta_adapter_target5_p3_s392002_20260825_r1.json`
- 复用release：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_p4_s392002_20260825_r2_scorerfix1/checkout`；其归档远端SHA256已核对为`467aef9c963b3842f3e5ccf89258fa8c4d0d198dd85f66b4a5e8fb687d02fc78`。
- 工厂输出root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_p3_s392002_20260825_r1`
- smoke output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p3_s392002_20260825_r1_smoke`
- prediction output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_p3_s392002_20260825_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_p3_s392002_20260825_r1.out`

## 验证、停止与晋级

- 当前生产代码已通过70项Stage2直接回归及199项Meta-Adapter Phase1／Phase2邻近回归；P4真实运行已验证相同runner的严格checkpoint加载、3步梯度更新、query冻结和truth-last评分闭合。
- P3仍先执行真实checkpoint无query smoke；技术停止仅限协议越权、错误row／split、输出覆盖、prediction不完整、scorer连接错误或确定性执行故障，不因性能停止。
- 15-row prediction完整后才连接truth。`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才晋级Target25，否则记录`SCIENTIFIC_FAILURE_NO_PROMOTION`并继续下一候选。
