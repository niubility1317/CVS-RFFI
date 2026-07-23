# ADV3B02全量CSIL与MoPC-HR新类无LEO配对诊断v1

Git承载镜像。完整预登记与最终结果应与根目录`automation_reports/CV-SincNet/adv3b02_paperfull_newclass_no_leo_20260723_v1/report.md`同步。

- 实验ID：`adv3b02_paperfull_newclass_no_leo_20260723_v1`
- 状态：`LOCAL_VERIFIED / NOT_LANDED`
- 证据标签：`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`
- 对照：`adv3b02_unfrozen_paperfull_ci_20260723_v7`
- 唯一实验变量：新类support/query由同物理记录LEO IQ替换为未叠加IQ；旧类LEO IQ不变。
- 矩阵：5receiver×5seed×4new count×4K×2method=800cell，三个旧类LEO切片共2400行。
- base-state：与v7相同且`base_sample_count=80`，因此本轮只隔离新类信道。
- 本地验证：`py_compile`PASS；focused pytest`2 passed`。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_paperfull_newclass_no_leo_20260723_v1`
- 最终结果：待N607完成后同步回填。
