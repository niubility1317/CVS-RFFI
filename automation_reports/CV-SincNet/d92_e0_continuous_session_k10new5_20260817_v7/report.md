# D92 E0连续session实验发布报告

## 运行身份与状态

- 实验ID：`d92_e0_continuous_session_k10new5_20260817_v7`
- 方法：`D92_E0_CUMULATIVE_REPLAY_SESSION_V1`
- 当前状态：`LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 科学代码提交：`75fcf33b9c1d77fbb1ba83bf555209944ccd6d32`
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
