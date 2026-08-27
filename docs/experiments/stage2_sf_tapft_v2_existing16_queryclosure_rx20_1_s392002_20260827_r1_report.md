# SF-TAPFT现有16个bundle独立query闭合实验

## 预登记

- run ID：`stage2_sf_tapft_v2_existing16_queryclosure_rx20_1_s392002_20260827_r1`
- 当前状态：`ANALYZED`
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

### 发布与运行闭合

- 固定提交：`a3405cd5f2468df1641c3e3ae48f5edd93df73c8`；提交已推送，发布前远端分支OID与本地`HEAD`一致。
- 单一release归档SHA256：`caeef91c15a9efe37a78f4a4a99fba1e831184d989eb07018ec217a84aebbf0b`；N607接收后同一归档SHA一致，远端编译通过。
- 真实checkpoint无query smoke：`PASS`；R0的`DA0_REG0/DA1_REG0`均严格加载为旧6类，`query_opened=false`。
- 运行：GPU0–7各2行，共16个独立prediction进程；启动后核对PID、cmdline、bundle、support/query、输出根、GPU归属和日志增长。
- prediction闭合：16/16份`prediction_receipt.json`、32/32份状态NPZ完整，日志无`Traceback/Error/Exception`。
- truth-last评分闭合：全部prediction完成后才由独立scorer打开`truth.npz`；16/16份`score.json`及总表完整，全部`same_row_ids=true`、`truth_join_after_prediction_only=true`。
- 版本兼容：R0按其原始clean-r16 release加载；M01–M15按capacity15 release加载。两族都只叠加本提交的query闭合模块，未改bundle内容或适配状态。

### 16行同row结果

单位：BA和floor为百分比；`ΔBA/Δfloor=DA1_REG0-DA0_REG0`，`ΔNLL<0`表示改善。该数据每类10条，因此单个query对应1.67pp总体BA、10pp类别准确率。

|候选|DA0 BA|DA1 BA|ΔBA(pp)|DA0 floor|DA1 floor|Δfloor(pp)|DA0 NLL|DA1 NLL|ΔNLL|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|R0|71.67|80.00|+8.33|10.00|30.00|+20.00|0.9367|0.5984|-0.3383|
|M01|71.67|81.67|+10.00|10.00|50.00|+40.00|0.9367|0.5731|-0.3636|
|M02|71.67|86.67|+15.00|10.00|60.00|+50.00|0.9367|0.5094|-0.4272|
|M03|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5347|-0.4020|
|M04|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5356|-0.4011|
|M05|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5354|-0.4013|
|M06|71.67|85.00|+13.33|10.00|50.00|+40.00|0.9367|0.5367|-0.4000|
|M07|68.33|81.67|+13.33|20.00|50.00|+30.00|1.1763|0.5659|-0.6104|
|M08|70.00|81.67|+11.67|30.00|50.00|+20.00|1.3426|0.5652|-0.7774|
|M09|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5406|-0.3961|
|M10|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5445|-0.3922|
|M11|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5438|-0.3929|
|M12|71.67|81.67|+10.00|10.00|50.00|+40.00|0.9367|0.5438|-0.3928|
|M13|71.67|81.67|+10.00|10.00|50.00|+40.00|0.9367|0.5483|-0.3883|
|M14|71.67|83.33|+11.67|10.00|60.00|+50.00|0.9367|0.5408|-0.3959|
|M15|71.67|81.67|+10.00|10.00|50.00|+40.00|0.9367|0.5439|-0.3928|

### DA1_REG0逐类准确率

|候选|类0|类1|类2|类3|类4|类5|
|---|---:|---:|---:|---:|---:|---:|
|R0|90|90|70|30|100|100|
|M01|80|90|70|50|100|100|
|M02|90|100|80|60|90|100|
|M03|90|90|80|60|80|100|
|M04|90|90|80|60|80|100|
|M05|90|90|80|60|80|100|
|M06|90|90|90|50|90|100|
|M07|90|90|80|50|80|100|
|M08|90|90|80|50|80|100|
|M09|90|90|80|60|80|100|
|M10|90|90|80|60|80|100|
|M11|90|90|80|60|80|100|
|M12|90|90|80|50|80|100|
|M13|90|90|80|50|80|100|
|M14|90|90|80|60|80|100|
|M15|90|90|80|50|80|100|

### 深入分析与结论

1. **独立query上域适应有效。**16/16行的BA、floor和NLL方向全部改善；R0也从80.00%以下的DA0基线提升到80.00%，说明此前仅在support内得到的信号并非完全由support重用造成。
2. **M02是本轮三指标共同最优。**M02的DA1 BA为86.67%、floor为60%、NLL为0.5094；相对自身DA0分别改善+15.00pp、+50pp和-0.4272。M06以85.00%排名第二，但只少1个正确query，当前样本量不足以把1.67pp差距解释为稳定优势。
3. **主要增益来自浅层可训练集合。**M01仅到head时为81.67%，加入norm的M02升至86.67%；继续开放P2/P3/P4及rank32并未继续提高，M03–M05都为83.33%。这支持下一阶段以M02为上界锚点，优先压缩而不是扩大encoder可训练范围。
4. **类别3仍是决定floor的瓶颈。**共同DA0中类3仅10%；M02把它提升到60%，同时类0/2/5分别改善20/10/20pp，但类4从100%降到90%。因此整体提升并不等于逐类无回退，后续确认必须继续保留floor和逐类表。
5. **rank继续减小出现边际损失。**P3下rank8与rank4均为81.67%且floor50%，rank4没有新增收益；P4下rank16为83.33%/floor60%，rank8降至81.67%/floor50%。当前证据不支持把rank8或rank4直接替代M02。
6. **rho变体不能只看ΔNLL。**M07/M08改变了DA0分类插值定义，DA0分别变为68.33%和70.00%，所以其大幅负`ΔNLL`不能与共同DA0候选直接作纯训练机制比较；最终DA1都只有81.67%。

### 科学判定与下一步

- 本轮结论：`SINGLE_SEED_SINGLE_RECEIVER_CLEAR_QUERY_CLOSURE_POSITIVE`。
- 当前首选：M02（`P1/head+norm`）进入下一轮最小确认；M06作为一条备选，因为BA只差1个query但其floor和NLL均弱于M02。
- 不晋级：M07/M08及rank4/8激进瘦身行；它们未超过M02的最终DA1表现。
- 不能宣称：尚不能外推到其他receiver、seed、`leo_low_elev_weak`、`leo_rain_weak`或完整Phase2；本实验是旧6类`REG0`，新类准确率、old/new harmonic和注册效应均为`N/A`。
- 建议下一步：先对M02做不同seed/receiver的最小同row确认；若保持优势，再围绕M02减少norm范围或冻结部分head参数，避免回到已被本轮否定的更深P2–P4路线。
