# ADV3B02全量CSIL与MoPC-HR新类无LEO配对诊断v1

Git承载镜像。完整预登记与最终结果应与根目录`automation_reports/CV-SincNet/adv3b02_paperfull_newclass_no_leo_20260723_v1/report.md`同步。

- 实验ID：`adv3b02_paperfull_newclass_no_leo_20260723_v1`
- 状态：`ANALYZED`
- 原始运行标签：`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`
- 永久解释标签：`LEGACY_ADAPTATION_CHANNEL_DIAGNOSTIC_NON_OFFICIAL`
- 对照：`adv3b02_unfrozen_paperfull_ci_20260723_v7`
- 唯一实验变量：新类support/query由同物理记录LEO IQ替换为未叠加IQ；旧类LEO IQ不变。
- 矩阵：5receiver×5seed×4new count×4K×2method=800cell，三个旧类LEO切片共2400行。
- base-state：与v7相同且`base_sample_count=80`，因此本轮只隔离新类信道。
- 本地验证：`py_compile`PASS；focused pytest`2 passed`。
- 本地Git commit：`9112401d7ac617f02b9ef959d6fa840addc28095`。
- runner SHA256：`516cef3874bf92281df963671f11ffd11a6f961f84ea8567f81bd5fb1d98f575`。
- focused test SHA256：`cbb190027bd6fda9e46a10b485854aa253c587ed722e6e537b754c9d41ed5066`。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_paperfull_newclass_no_leo_20260723_v1`
- N607完成：8/8分片、800/800cell、800/800prediction、2400/2400唯一结果行；全部prediction SHA匹配，完整性PASS。
- 结束状态：8张GPU全部释放，本地`ssh.exe=0`，N607/桥接TCP22连接均为0。
- 证据包SHA256：`7a0fb8baeb342e4ccd1012d33f2213fe3973bd9a6956caa5b55f31e958daac69`。
- 总体无LEO结果：
  - CSIL旧适配：旧类注册前/后`77.13/39.34%`，新类`2.75%`，H`3.44%`，遗忘`40.35%`，min-old`4.50%`。
  - MoPC-HR旧适配：旧类注册前/后`77.13/1.68%`，新类`40.09%`，H`2.53%`，遗忘`75.48%`，min-old`0.00%`。
- 相对v7同row：CSIL旧类后`+4.58pp`、新类`-2.52pp`、H`-2.79pp`；MoPC-HR旧类后`-1.06pp`、新类`+5.90pp`、H`-0.96pp`。
- 结论：去除新类LEO没有修复联合性能，信道不是整体崩塌主因。CSIL EWC与MoPC-HR HR均比分类loss高多个数量级；全量可训练ADV3B02与仅80条base-state是更强失败信号。
- 声明边界：本run使用后来确认的旧适配实现，只能作为`LEGACY_ADAPTATION_CHANNEL_DIAGNOSTIC_NON_OFFICIAL`信道消融，不能代表官方CSIL或MoPC-HR，不能进入论文正式对比或方法晋级。
- 完整根报告与artifact：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_paperfull_newclass_no_leo_20260723_v1\report.md`。
