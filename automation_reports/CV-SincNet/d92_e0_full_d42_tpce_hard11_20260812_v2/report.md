# D92 E0 FULL D42 TPCE Hard11 v2发布预注册

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tpce_hard11_20260812_v2`|
|目标|10个最难performance outer上相对E0_FULL_ONLY八指标严格Pareto筛选；另含1个K1 liveness|
|候选|`E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE`；`d92_e0_full_d42_tail_pair_code_exchange`|
|科学commit|`4768c6b2613ba3764bb9fbee95c9a7ce3561220f`|
|矩阵|11 jobs；33 scene-arm；8 shards；`p2_min_v1`|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|独立复核|递归修复后P0=0，P1=0，APPROVE|

v1在K10 truth-free smoke的manifest artifact校验中发生确定性递归，尚未进入TPCE fit、未生成active/fallback字段、未启动shard，已标`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v2只修复原始artifact validator的冻结引用；方法、矩阵、阈值、路径语义均未变。

发布输入：runtime archive`d92_tpce_runtime_closure_4768c6b2.tar.gz`，5,095,342B，SHA256=`e775b66584a83756e11834dfcf2908398c9707b783e2c1fe71628f6944f388a7`；method lock SHA256=`58dabf7ed4510c74aa2beff4031a2bbe745be940d2dc1b8361300ecf07f7f23c`；launch 3,764B，SHA256=`d8ef3c0ecd73bcc8b69671030e8d44fdb2daf82500a897855d7bbcdcd00e0ed2`。递归专项与TPCE机械聚焦16项通过，生产模块`py_compile`通过。

远端source/output/log分别为：

- `/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_4768c6b2_20260812_v2`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tpce_hard11_20260812_v2`
- `/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tpce_hard11_20260812_v2`

唯一启动命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tpce_source_snapshot_4768c6b2_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

真实K10 checkpoint smoke仍是8shard前置硬门：三场景必须TPCE active、无fallback、fit=2/1、D42 state/prediction closure闭合、所有query访问为false。`fresh_run_retry=false`；仅按协议/技术健康规则停止，绝不按性能停止。

## N607执行结果

唯一launch完成prepare（11 jobs、33 scene-arm，matrix manifest SHA256=`e2b4cda61881bdd7f8cfd497515a6f5706aa3a22fd95b8c1b24a996e64656c90`），随后在K10真实checkpoint truth-free smoke的TPCE激活门停止。三场景均为`active=false`、`fallback_active=true`、`fallback_reason=support_guard_failed`、`support_guard_pass=false`；未启动任何shard，未读取score或准确率，未重试。

|场景|6个旧类Q20 tail增益|pooled-new增益|hinge变化|guard tolerance|
|---|---|---:|---:|---:|
|clear|`[0.028202,-0.060852,0.021923,0.004051,-0.018242,0.011291]`|`0.040862`|`0.0`|`0.000806087`|
|low_elev|`[0.052156,-0.053238,0.041025,-0.001877,0.018299,0.015153]`|`0.028333`|`0.0`|`0.000758791`|
|rain|`[0.038380,-0.052732,0.016371,-0.005156,0.034668,0.011009]`|`0.030264`|`0.0`|`0.000681875`|

三场景共同证据表明：新类方向均为正、双向hinge未恶化且无饱和，但66个relation×block交换同步叠加后，至少一个旧类tail显著下降。因此失败原因不是D42分辨率不足或新类目标错误，而是全量原子交换之间的旧类干扰。下一修订只允许从同一固定原子集合中选取逐步保持六旧类tail、pooled-new、all-class和双向hinge安全的确定性Pareto子集；不增加fit、不扫描强度、不接触query。

## 闭环与取回

- 技术停止异常：`D92D92TPCEHard11RunnerError: fit audit TPCE K>2 candidate did not activate`。
- 正式计数：0/8 shards、0/11 formal jobs、0 formal scores；本次没有性能结果。
- source/log/output及11个manifest引用truth sidecar已完整取回到`E:/type10-7/local_artifacts/d92_e0_full_d42_tpce_hard11_20260812_v2`，sidecar逐SHA匹配；远端输出保留。
- 查询协议字段仍全部为false：query fit/update/selection/truth/role/quota/global reassignment均未发生。
