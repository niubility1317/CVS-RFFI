# CVS恢复交接（2026-09-28）

Phase1已完成目标评分，详见[完整结果](CVS_PHASE1_TARGET_RESULTS_20260928.md)。5个模型使用原始DAOT A1＋FastTrust-RC4的第200轮权重；Practical LEO residual/post_sync/noeq。不得据新目标分数调参或选择性重跑。

D92版本为P2-256-FULL，即E0去RF32、identity160＋FFT96。两次失败均为inner support LDA的FP32舍入错误；修复仅在原guard失败时用FP64消除类别共同仿射项后转换，严格sklearn预测检查保留，成功路径数值不变。修复、聚焦测试和一次独立审查完成。

当前run：`20260928-phase2-cvs-d92-practical-manytx-m5-r02`。release：`cvs_d92_recovery_20260928_r02`，commit：`c0e538d60c2e52a50112ee9b43337b4277d463d3`。远端dispatcher PID2061061已从/proc核实。原run、权重、日志和预测全部保留。3个完整seed在新run通过逐记录校验后原样复用；两个partial成功前缀也保留，只补算缺失划分。

实际运行和最新证据见`automation_reports/CV-SincNet/20260928-phase2-cvs-d92-practical-manytx-m5-r02/`。两个原失败split在远端完整support-only回放均FIT_PASSED。初次运行核查时392005为850/2121、2026092703为1990/2121；这只是历史进度，后续不得据此推断实时状态。

在Git工作树`E:/type10-7/code/snapshots/daot_practical_three_20260918_wt`下运行已验证Python：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/read_cvs_d92_recovery.py
```

这是只读核查，不启动新任务。不要重复调用publisher。5行均达到2100 splits和2121条预测后，dispatcher自动独立评分，产物为远端run/scored_results.json和completion.json；之后下载结果、做逐seed/scene/K/new_count统计、更新原记录并提交。发生异常则先保留partial并定位，不按性能停机或更换seed。唯一launch owner为codex/root/cvs-d92-recovery-20260928，未使用GPU训练名额。
