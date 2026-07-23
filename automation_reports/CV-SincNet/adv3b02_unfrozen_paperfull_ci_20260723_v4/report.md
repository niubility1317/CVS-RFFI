# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v4

- 类型：`FORMAL_PAPER_METHOD_COMPARISON_BASELINE`
- 状态：`PREREGISTERED_REMOTE_BASE_BUILDER_COMPAT_LOCAL_VERIFIED_SMOKE_PENDING`
- ADV3B02不冻结；CSIL与MoPC-HR参数和v3保持不变；新类support/query必须为LEO后信道IQ。
- comparison-only wrapper保留验证后的原始sample ID，并兼容禁用远端旧`_assert_scenario_alignment`与本地新`_assert_scenario_physical_independence`两个Stage2主方法门禁；通用builder未修改。
- legacy LEO的17成员、文件SHA、逐样本IQ/overlay SHA及顺序根验证保持不变。
- builder SHA=`5242c17e0c277f22e4af4df3be89ebdd9cbe306ab0f64023eb220b4ae0e0f748`；30项focused test、`py_compile`、`git diff --check`PASS。
- 新远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v4`；固定4cell smoke全链PASS后才授权100 package/800cell/2400行正式矩阵。

完整预登记和停止条件见工作区主报告：
`E:\type10-7\automation_reports\CV-SincNet\adv3b02_unfrozen_paperfull_ci_20260723_v4\report.md`。
