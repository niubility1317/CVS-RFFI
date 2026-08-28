# SF-TAPFT P1 D3 R1-T定点修复实验报告

## 1.最小预登记

- run ID：`stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- 当前状态：`LOCAL_VERIFIED`
- 实现commit：`24493929ffce87f371bf036b219cc733b75d05eb`
- 候选：D3 R1-T；327/0/0固定步；all time norm；4-fold support-only OOF温度；原位full-support refit；delta v2 only
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`；旧6类K=10，共60条support
- query边界：OOF温度和full-support refit只读support，不读取query、truth、role或source；本run只产生适配delta
- 触发原因：原矩阵D3虽配置OOF温度，但旧deploy入口未执行；旧D3标记`METHOD_MISMATCH_NO_R1T_RESULT`并保留，不覆盖

## 2.验证、命令与路径

- 本地`ssr-gpu`相关测试117项通过；D3定点P0/P1复审PASS
- N607环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch2.1.0+cu121
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`
- GPU：0
- 命令：`python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json --row-id D3 --mode deploy --deployment-inplace --delta-only --output-dir <run-root>/support/D3 --device cuda:0`
- 预期artifact：`selection.json`、`sf_tapft_delta_bundle.pt`、stdout/stderr、GNU time
- 停止规则：仅协议/query泄漏、错误数据句柄、输出碰撞、错误checkout、确定性异常、无合法delta或进程归属不清；不得因低性能停止

## 3.最终闭合结果

- 状态：`ANALYZED_ENGINEERING_REPLAY/NO_PROMOTION`；
- support适配：`DEPLOY_ADAPT_COMPLETE`；prediction先于truth完整形成，score记录`truth_join_after_prediction_only=true`；
- OOF温度：4-fold support-only拟合，`T=1.198155`，OOF NLL从0.665581降到0.646321，argmax保持不变；
- 最终head scale：6.676930；
- Q180共同`DA0_REG0`：130/180，BA 72.2222%，floor 10.0000%，NLL 0.870038，ECE-10 0.130062；
- Q180 `DA1_REG0`：154/180，BA 85.5556%，floor 63.3333%，NLL 0.575789，ECE-10 0.082368；
- DA效应：BA +13.3333pp，floor +53.3333pp，NLL -0.294249，ECE-10 -0.047694；
- 同row配对：错→对31、对→错7、都对123、都错19，精确McNemar `p=0.000116`。

## 4.逐类结果与误差结构

|类别|正确数/30|准确率|NLL|
|---:|---:|---:|---:|
|0|29|96.6667%|0.439523|
|1|28|93.3333%|0.291341|
|2|27|90.0000%|0.388583|
|3|19|63.3333%|1.249931|
|4|24|80.0000%|0.645451|
|5|27|90.0000%|0.439905|

混淆矩阵如下，行是真实类，列是预测类：

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|---:|
|0|29|1|0|0|0|0|
|1|2|28|0|0|0|0|
|2|0|1|27|0|0|2|
|3|4|6|0|19|1|0|
|4|0|5|0|1|24|0|
|5|0|0|0|1|2|27|

D3相对修复后D0把类0提高16.6667pp、类3提高6.6667pp，但类4回退10pp。类4NLL升至0.645451，说明全局温度虽改善support OOF校准，却不能修复局部类别边界。

## 5.分区与资源

|分区|BA|floor|NLL|
|---|---:|---:|---:|
|Q60|83.3333%|60.0000%|0.570450|
|Q120|86.6667%|65.0000%|0.578459|
|Q180|85.5556%|63.3333%|0.575789|

- 可训练/实际变化元素：1584/1584；
- 固定训练：327步full-support refit并附加4-fold OOF；
- 完整backbone forward：327次，另有OOF训练计算；
- raw snapshot：6336B；delta v2：8211B；
- GNU time墙钟：58.45秒；最大RSS：1809016KiB；
- GPU显存峰值：`NOT_CAPTURED`，不作推断。

## 6.晋级判定

相对修复后D0，D3通过BA、floor、可训练元素和delta大小门槛，但失败于三项：

- 类4相对D0回退10pp，低于-5pp类别保护线；
- NLL 0.575789，高于`D0+0.02=0.520754`；
- 58.45秒高于20秒适配墙钟上限。

因此D3只保留为绝对BA/floor最高的性能研究档，不替代D0部署档。由于rx20 query truth已有历史暴露，本结论仅是工程回放，不构成正式科学晋级。
