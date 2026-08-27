# SF-TAPFT F2最大独立Query适应前后对比

## 预登记

- run ID：`stage2_sf_tapft_f2_maxquery120_rx20_1_s392002_20260827_r1`。
- 科学问题：在F2+OOF温度的同一SF-TAPFT clean-single bundle上，将旧6类真实Query从旧support pool内部留出的60条扩大为母资产中全部独立Query 120条，比较`DA0_REG0`与`DA1_REG0`。
- 候选：F2，adapter rank=16，A/B/C步数=`100/100/0`，OOF温度=`1.071440445141165`；不注册新类。
- 数据：receiver=`20-1`，scene=`leo_clear_weak`，旧6类；support固定K=10×6=60条；Query使用既有预测包的完整独立`query_leo_clear_weak.npz`，120条固定received-IQ。母资产登记为每类20条独立Query，support与Query物理ID不重叠，`p2_min_v1/VALIDATED_ONCE`。
- 对比状态：`DA0_REG0`与`DA1_REG0`；两臂使用完全相同的120个opaque Query ID。新类与REG1指标均为N/A。
- 代码提交：`0f3152b4e4b3c1f098fc9c51199dba02140c657c`。
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`；Conda环境`CVS-RFFI`。
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1/F2/output/sf_tapft_clean_single_bundle.pt`。
- 输入母包：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase2_predictor_packages/rx_20_1/seed_713101`。
- scorer truth sidecar：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase2_scoring_sidecars/rx_20_1/seed_713101/truth_sidecar.json`；仅在两份prediction完成并通过几何、有限性与同row检查后连接。
- 远端输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_f2_maxquery120_rx20_1_s392002_20260827_r1`，不可覆盖。
- GPU：0。
- 预计artifact：`inputs/{support.npz,query.npz,data_handle.json}`、`prediction/{da0_reg0.npz,da1_reg0.npz,prediction_receipt.json}`、`truth_after_prediction.npz`、`score.json`、GNU time与GPU采样。
- 执行顺序：真实bundle无Query smoke→无truth导出120条Query→`DA0_REG0/DA1_REG0`预测→完整性检查→连接truth sidecar→独立评分。
- 技术停止规则：仅在协议/query泄漏、错误receiver/scene/K/split、输出碰撞、bundle绑定失败、prediction不完整、同一确定性异常或进程归属不清时停止；不得因性能低而停止。
- 晋级/解释：本run用于提高F2适应前后估计精度，不改变F2选择或温度；完整报告将给出总体BA、floor、NLL、各类正确数/20与准确率，并与上一轮60条Query结果作样本口径区分。
