# SF-TAPFT现有16个bundle独立query闭合实验

## 预登记

- run ID：`stage2_sf_tapft_v2_existing16_queryclosure_rx20_1_s392002_20260827_r1`
- 当前状态：`LOCAL_VERIFIED`
- 候选：现有`R0+M01–M15`共16个`sf_tapft_clean_single_bundle.pt`，不重训、不修改bundle。
- 数据：`p2_min_v1/VALIDATED_ONCE`；receiver=`20-1`；场景=`leo_clear_weak`；旧6类K=10 support共60条；独立query为同一已验证pool的rank10–19，旧6类各10条、共60条。
- query capsule：`sf-erbt-oldonly-rx20-1-s713101-clear-k10-holdout10-v1`。
- query split：`p2_min_v1-rx20-1-s713101-clear-old6-k10-rank0_9-holdout-rank10_19`。
- 物理边界：support/query物理ID交集为0；query NPZ只含`received_iq/query_ids`；truth单独位于`truth.npz`，仅在全部prediction完整后由独立scorer打开。
- 状态：每个候选同row比较`DA0_REG0`和`DA1_REG0`；REG0新类指标为`N/A`。
- 指标：accuracy、balanced accuracy、class floor、NLL、逐类准确率，以及`DA1_REG0-DA0_REG0`差值。
- GPU：16行按每张GPU两行分配到GPU0–7；不超过每卡两个实验。
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- N607数据根：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`。
- N607输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v2_existing16_queryclosure_rx20_1_s392002_20260827_r1`。
- N607日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v2_existing16_queryclosure_rx20_1_s392002_20260827_r1`。
- 预期artifact：每行`prediction/{da0_reg0.npz,da1_reg0.npz,prediction_receipt.json}`和truth-last`score.json`，矩阵完成后生成汇总表。
- 技术停止：仅限错误bundle/data绑定、query truth/role进入predictor、support/query重叠、错误checkout、输出碰撞、无法加载真实checkpoint、无prediction闭合或确定性系统异常；不得因低性能停止。
- 科学判断：先按独立query的`DA1_REG0`balanced accuracy排序；瘦身候选必须同时报告相对自身`DA0_REG0`的均值、floor和NLL变化。单seed单receiver单场景只用于现有16个bundle的首轮闭合与下一步筛选，不外推完整Phase2。

## 本地验证

- 新增truth-blind预测与truth-last评分入口；聚焦与邻近回归18项通过。
- 两个Python入口静态编译通过。
- P0/P1正确性检查：query成员白名单在bundle加载前执行；support字节摘要、K10平衡性和support/query ID零交集在预测前复核；输出不可覆盖；预测artifact不含truth/role；评分严格连接同一60个query ID。未发现会导致本次实验跑错、越权、覆盖输出、不能启动或不能产生合法prediction的P0/P1。

## 完成结果

待prediction与独立评分完成后填写。
