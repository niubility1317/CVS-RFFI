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
