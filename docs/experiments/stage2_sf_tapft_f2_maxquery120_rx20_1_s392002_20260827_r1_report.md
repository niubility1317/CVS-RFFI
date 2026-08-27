# SF-TAPFT F2最大独立Query适应前后对比

## 预登记

- run ID：`stage2_sf_tapft_f2_maxquery120_rx20_1_s392002_20260827_r1`。
- 科学问题：在F2+OOF温度的同一SF-TAPFT clean-single bundle上，将旧6类真实Query从旧support pool内部留出的60条扩大为母资产中全部独立Query 120条，比较`DA0_REG0`与`DA1_REG0`。
- 候选：F2=`S15-SCHED300+OOF全局温度`；完整cosine日程300步、warmup 30步，研究选择步为`[150,0,0]`，训练`head+all time norm`且不训练Adapter；OOF温度=`1.071440445141165`；不注册新类。
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


## 最终状态：ANALYZED

严格smoke回读纠正了预登记中的一处方法元数据：实际F2 bundle为`S15-SCHED300+OOF全局温度`，完整cosine日程300步、warmup 30步，研究选择步为`[150,0,0]`，训练`head+all time norm`且不训练Adapter；可训练/实际变化元素均为1584。预登记写成`100/100/0`属于报告记录错误，真实bundle、预测命令和输出均未受该文本影响。

独立Query分区120条已完成truth-last评分：

|指标|`DA0_REG0`|`DA1_REG0`|域适应效果|
|---|---:|---:|---:|
|正确数|87/120|100/120|+13|
|BA/总体准确率|72.5000%|83.3333%|+10.8333pp|
|最低类别准确率|10.0000%|50.0000%|+40.0000pp|
|NLL|0.836713|0.530394|-0.306319|
|ECE-10|0.118572|0.064598|-0.053974|

各类正确数/20依次为：适应前`11/20,20/20,19/20,2/20,19/20,16/20`；适应后`18/20,20/20,19/20,10/20,16/20,17/20`。同row配对中16条由错转对、3条由对转错、84条始终正确、17条始终错误；双侧精确McNemar检验`p=0.004425`。

prediction阶段墙钟6.94秒，最大RSS为1,150,280KiB。进程在连续GPU采样前已结束，因此不声明GPU显存峰值。完整120条结果属于最大独立Query分区；将它与上一轮零重叠的60条rank10–19 holdout合并后，可覆盖现有固定IQ中除K10 support外的全部180条Query，见单独的最大并集报告。
