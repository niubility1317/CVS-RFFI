# SF-TAPFT-PACE E0–E3实验报告

## 预登记

- run ID：`stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828_r1`。
- 科学问题：在D0 Compact之后追加120步受保护time-Norm容量升级，并用support cross-fitted head logits拟合6参数零和类别bias，能否在一分钟预算内提高旧6类`DA1_REG0`性能且抑制局部负迁移。
- 矩阵：E0=`D0 Compact`；E1=`PACE-W`（新增`t2.norm.weight`）；E2=`PACE-All`（新增`t2/t1/time_fuse`的weight）；E3=`PACE-All+40步bias-only OOF`。
- 固定损失：`lambda_tail=0.03`、`lambda_preserve=0.10`；E1–E3扩展120步；禁止HardPair、Adapter、完整t3、frequency/domain更新和EMA。
- 数据：`p2_min_v1/VALIDATED_ONCE`；receiver=`20-1`；scene=`leo_clear_weak`；seed=`713102`；旧6类K=10，共60条support；独立Query每类20条，共120条。该seed在本轮前未形成SF-TAPFT评分记录，prediction前不连接truth。
- adaptation capsule：`d18-enrollment-before-rx20-1-seed713102-k10-pace`；split：`stage2b-rx20-1-seed713102-before-support-prefix`。
- query capsule：`sf-tapft-pace-rx20-1-s713102-clear-k10-independent20-v1`；split：`p2_min_v1-rx20-1-s713102-clear-old6-k10-independent20`。
- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- Git提交：`f83154f3b503cb012574be36db1bdf51dc65ba7b`（共享工作树中的并行ERBT任务提交时一并纳入本轮已暂存PACE文件；文件内容与51项本地验证一致）。
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`；Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 本地配置：`configs/stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828.json`。
- 远端输入：`runs/stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828_r1/inputs`；输出：同run根目录下`E0`–`E3`及`query/<row>`，均不可覆盖。
- GPU：E0/E1/E2/E3分别为GPU0/1/2/3。
- 启动命令：每行使用`code/scripts/run_sf_tapft_slim_matrix_row.py --mode deploy --deployment-inplace --delta-only`；完成delta后使用truth-blind query闭合器产生`DA0_REG0/DA1_REG0`prediction，最后由独立scorer连接truth。
- 预期artifact：各行`selection.json`、`sf_tapft_delta_bundle.pt`、GNU time与GPU采样；各行Query的两状态prediction、prediction receipt、truth-after-prediction和`score.json`。
- 技术停止规则：仅协议/query泄漏、错误receiver/scene/K/split、输出覆盖、错误checkout、无prediction闭合、scorer连接错误、进程归属不清或同一确定性预prediction异常可停止；不得因低性能停止。
- 晋级门槛：相对E0，BA不降、floor不降、最差类别变化不低于-5pp、NLL不高于E0+0.02；warm-resident中位数不超过60秒；delta不超过16KB。

## 当前状态

`LOCAL_VERIFIED`。PACE聚焦及部署回归共51项通过；N607只读preflight通过，GPU0–7均空闲。新seed=`713102`的predictor package、独立120条Query和detached truth sidecar存在；尚未连接truth。
