# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v7`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 当前状态：`ANALYZED / DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN / REJECT_RESOURCE`
- 科学代码提交：`75fcf33b9c1d77fbb1ba83bf555209944ccd6d32`
- 本地truth-last重定位修复提交：`2ba7b835`
- 目标：直接完成5个outer的连续单类/少数类注册性能、资源与终端等价实验。

## 本轮唯一修复

v6的smoke只冻结`batch_5`和`singleton_forward`两条liveness轨迹，预测层却错误要求smoke同时含四条正式轨迹，因而在Phase B报`terminal schedule closure drift`。v7让smoke严格比较它实际运行的两条轨迹；正式run仍传入并严格比较全部四条冻结轨迹。方法、数据、阈值、查询身份、状态编译和评分规则均未改变。连续session相关测试52项通过。

## 冻结实验

| 维度 | 值 |
|---|---|
| outer / seed / K | `20-1`,`3-19`,`7-14`,`7-7`,`8-8` / `713106` / `10` |
| scene | `leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak` |
| schedule | `batch_5`、`singleton_forward`、`singleton_reverse`、`chunk_2_2_1` |
| 规模 | 5 outer×3 scene×4 schedule；210次`DA1_REG1`注册 |
| 注册裁决 | wall≤300ms、增量working-set≤4MiB；超限不阻断，truth-last分析标`REJECT_RESOURCE` |
| 实时推理 | 全注册类独立判决；零query访问/更新；state与`C×288`MAC闭合 |

## N607交接

- CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v7`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_continuous_session_k10new5_20260817_v7`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU0 smoke；正式5 outer固定GPU0–GPU4。

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_continuous_session_k10new5_20260817_v7 && nohup bash ./launch.sh >./launch.out 2>./launch.err </dev/null &
```

只允许单次启动；runner不读性能、不运行analyzer、不调参、不重试。技术健康完成后才取回5份truth sidecar，由主代理运行truth-last分析。

## Runner完成记录（2026-08-17）

- 技术终态：`ARTIFACTS_COMPLETE / READY_FOR_PRIMARY_ANALYZER`。这是技术产物完整状态，不是性能结论。
- 落地：archive为171137B、SHA256=`791b1eb3f243b93371cec22f3bb5aabace8e6de57fb4a94f2890d1b9087e1054`、31成员且安全tar验证通过；launch为2261B、SHA256=`6845226173379b07cf763d0dd9e1ff888a34d24d0361304189fc7dd2e7597775`且`bash -n`通过。冻结启动命令仅提交一次。
- 连接边界：启动SSH客户端未正常退出；仅识别并处理本次启动边界对应的本地PID 32276。正常结束失败后才强制结束该单一客户端；后续远端只读核验确认run已落地。结束时本地`ssh.exe`、`scp.exe`和至N607 TCP22连接均为0。
- 健康与闭合：无run-owned PID、GPU compute进程或技术异常标记。smoke严格比较`batch_5`与`singleton_forward`两条轨迹；正式5个outer均严格比较四条冻结轨迹。所有已核验closure状态为`STRICT_EQUAL`；query truth/fit/update/selection/role/quota/global-reassignment访问均为false。
- 已保全产物：`matrix_manifest=1`、`prepared_manifest=5`、`delta_receipt=5`、`prediction_manifest=6`（smoke+5 formal）、`job_receipt=5`、`execution_receipt=246`、`fit_audit=246`、`resource_audit=246`、`COMMIT=251`。
- 取回：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v7/run_root`（1420文件）及`log_root`（10文件）已完整取回。5份manifest-bound `truth_sidecar.json`已独立复制到`truth_sidecars/jobs/<outer>/offline/scorer/`；5份`job_receipt.json`随run root保全。truth内容未读取，`content_read=false`。
- 边界：runner未读取accuracy、H、BA、floor或其他性能字段，未运行analyzer、未作资源或性能解释。现在仅由主代理进行truth-last分析。

## 主代理truth-last分析（2026-08-17）

### 分析闭合

- 主代理先完整校验5个prediction manifest、246组prediction/fit/resource/execution/COMMIT绑定，再打开5份truth sidecar；`prediction_validation_complete_before_truth_open=true`，`truth_sidecar_exposed_to_predictor=false`。
- 取回后的manifest保留N607绝对路径，原分析器首次本地执行报`prediction manifest location drift`。提交`2ba7b835`增加显式本地重定位，只改变离线artifact定位，不修改预测、truth、状态、资源收据或评分规则；连续session相关测试`54 passed`，Python编译与`git diff --check`通过。N607未重跑。
- 权威分析artifact：`E:/type10-7/local_artifacts/d92_e0_continuous_session_k10new5_20260817_v7/analysis_v2/continuous_session_analysis.json`，955368B，SHA256=`0d7cdcb2961bc5d46048677e30d5042315f89be02cf2c92dbcd05cf67f4de362`。
- 分析覆盖5 outer×3 scene。`singleton_forward`和`singleton_reverse`均按`1+1+1+1+1`注册；每个session新增1类、20条已注册新类query，已评分新类query数依次为20/40/60/80/100，尚未注册的80/60/40/20/0条query明确标为`UNREGISTERED_NOT_SCORED`，没有把未注册类按零分计入新类准确率。

### 逐类连续注册主结果

下表是`singleton_forward`在15个outer×scene同排表面上的聚合。`均值/最差`均来自同一session状态；遗忘为`DA1_REG0 old BA-当前old BA`，正值表示遗忘。

|状态|已注册新类数|有效新类query|H均值/最差|old BA均值/最差|old floor均值/最差|seen-new均值/最差|平均遗忘|new→old|old→new|资源通过|注册wall均值/最大|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`DA1_REG0`|0|N/A|N/A|84.89%/72.50%|64.00%/35.00%|N/A|0.00pp|N/A|0.00%|N/A|N/A|
|`DA1_REG1_S1`|1|20|71.93%/45.70%|80.61%/65.83%|54.67%/35.00%|67.00%/35.00%|4.28pp|33.00%|4.50%|0/15|1610.65/2892.36ms|
|`DA1_REG1_S2`|2|40|75.22%/49.76%|79.94%/65.83%|53.00%/30.00%|72.17%/40.00%|4.94pp|24.33%|7.00%|0/15|1868.15/2742.64ms|
|`DA1_REG1_S3`|3|60|74.23%/47.99%|78.39%/63.33%|51.67%/25.00%|71.67%/38.33%|6.50pp|21.22%|9.94%|0/15|1911.21/3240.33ms|
|`DA1_REG1_S4`|4|80|73.63%/42.57%|77.17%/61.67%|49.33%/25.00%|71.75%/32.50%|7.72pp|19.08%|11.00%|0/15|2146.61/3051.51ms|
|`DA1_REG1_S5`|5|100|74.53%/48.99%|76.56%/60.83%|49.67%/15.00%|73.33%/41.00%|8.33pp|17.93%|13.33%|0/15|1821.07/2540.52ms|

连续注册不是单调改善。S1→S2的H和新类准确率上升，但S3/S4旧类BA、floor和H继续下降；S5虽恢复部分新类准确率，旧类平均遗忘已达8.33pp，最差old floor降至15.00%。反向顺序S1的H均值为83.38%，明显高于正向S1的71.93%，说明中间session对新类到达身份与顺序敏感；这不影响终端等价结论，但否定了“任意单类到达都具有同等中间质量”。

### 终端同排结果

四条轨迹在每个outer×scene的最终状态、预测artifact和八项指标均严格相等，共`15/15 STRICT_EQUAL`。因此`batch_5`、正向逐类、反向逐类和`2+2+1`在全部5类注册完成后得到同一终态；下表按outer聚合3个scene，所有数值保持同一终端行语境。

|outer|K/seed|H|old BA|old floor|seen-new|平均遗忘|new→old|old→new|终端等价|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`20-1`|10/713106|80.24%|81.94%|65.00%|78.67%|6.94pp|12.33%|12.78%|3/3|性能较稳；资源拒绝|
|`3-19`|10/713106|51.31%|61.67%|26.67%|44.00%|13.61pp|32.33%|21.67%|3/3|最弱outer；资源拒绝|
|`7-14`|10/713106|79.32%|76.94%|46.67%|82.00%|7.50pp|12.33%|11.11%|3/3|floor偏低；资源拒绝|
|`7-7`|10/713106|83.42%|85.83%|63.33%|81.33%|8.61pp|17.00%|9.44%|3/3|H最高；资源拒绝|
|`8-8`|10/713106|78.35%|76.39%|46.67%|80.67%|5.00pp|15.67%|11.67%|3/3|floor偏低；资源拒绝|

按scene聚合5个outer，`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`的终端H分别为79.08%/71.11%/73.39%，seen-new分别为79.00%/68.40%/72.60%。最差H同排为`3-19×leo_low_elev_weak`：H=48.99%、old BA=60.83%、old floor=40.00%、seen-new=41.00%、遗忘=14.17pp；最差floor同排为`3-19×leo_clear_weak`：old floor=15.00%、H=53.03%、seen-new=47.00%。

### 资源与最终裁决

|资源项|冻结门|观测|结果|
|---|---:|---:|---|
|注册状态数|210|210|闭合|
|注册wall|≤300ms|最小77.89ms；均值1802.45ms；最大3443.22ms|202/210失败|
|增量working-set|≤4MiB|最大3.89MiB|210/210未越过硬上限|
|综合资源门|全部状态通过|8/210通过|`REJECT_RESOURCE`|

实验技术上已完成，逐session性能也已形成truth-last结果；但该候选不能晋级。原因不是预测闭合或协议失败，而是注册wall资源门大面积失败，同时`3-19`outer和旧类floor暴露出明显稳定性不足。最终标签为`DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN / REJECT_RESOURCE / NOT_PROMOTABLE`。本结果不替代完整Target125确认矩阵，也不声明真实在轨性能。
