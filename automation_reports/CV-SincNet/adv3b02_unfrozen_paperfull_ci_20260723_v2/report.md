# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v2

- 类型：`FORMAL_PAPER_METHOD_COMPARISON_BASELINE`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 代码提交：`f8a6195e`；comparison bundle修复：`22e44717`。
- v1在训练前因旧主方法cache-set门禁失败，0cell、无性能结果，远端只读封存；v2使用全新run root，禁止拼接。
- v2矩阵：100 package、800cell、2400个三LEO场景正式行。
- 唯一强制项目数据条件：全部新类注册support与新类评测query叠加LEO星地信道。base/source统计访问及完整论文资源对对比方法开放。
- comparison builder SHA：`59f37c6caa7ab10864c717d6e8ff3d56af6e58966bb6bae618791a6be9e6f13e`。
- runner SHA：`f4a632a67552cdcedd6f1435a0b79dec8967f7fc9be354f9325a5032e18e236a`。
- 远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v2`。
- v2 smoke PID=`864221`，在首个package训练前因旧inner cache缺少`source_dataset_sha256`与`source_record_indices`失败；package/cell/prediction/scoring均为0，无性能结果，正式矩阵未授权。GPU已释放，远端只读封存。

完整预登记、参数锁、失败隔离、矩阵、N607路径和停止条件见工作区主报告：
`E:\type10-7\automation_reports\CV-SincNet\adv3b02_unfrozen_paperfull_ci_20260723_v2\report.md`。
