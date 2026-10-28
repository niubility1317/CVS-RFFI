# PAMF Phase3：源域物理分支拒识结果

2026-10-29。**16条事前固定规则全部未达源侧95%拒识/5%误拒/2pp旧类保护联合门槛，NO_PROMOTION。** 不扩三fold或正式target；这些是源代理结果，不能代替完整六类PAMF在星地目标域的结果。

## 实际运行

run：`20261029-phase3-source-branch-s2026092701-r01`。执行code/config commit：`df34171b8dc9942ec236e4504cac61768d70997a`；N607 GPU2，一条导出worker、一个CPU规则进程，未停止其他owner。复用自身四类scratch CE20/540的固定源checkpoint，新增模型更新0、target读取0、默认头改动0。两条合法L真实checkpoint smoke PASS，之后导出L3360/V27000共30360条、8分支；全部导出固定后再运行16规则。流水线和两个完成标记独立读回，状态VERIFIED/ANALYZED_SOURCE_ONLY；全日志无错误或非有限数异常。

source split392005，model seed2026092701。模型训练TX为source labels2/3/4/5，RX1/3/4/6；labels0/1未参与该scratch模型，RX8未参与参数拟合。V中已知18000/代理未知9000，RX8切片已知3600/代理未知1800。所有规则共用完整默认头，闭集V已知92.2167%、RX8为66.4722%。这不是当前完整Tri-SIG-TIME SSL训练配方的四类复现。

## 固定16规则

除fused160沿用原L2基线，其余分支仅L拟合全局mean/std再L2；中心和共享类内收缩协方差只拟合L。V已知固定99%保留全局阈值；V未知只用于源开发评价。RX8 V参与了已知阈值校准，因此该接收机只从模型拟合留出，不能声称严格全流程未见接收机确认。

|规则|全V未知拒识|RX8未知拒识|RX8旧类误拒|RX8含误拒旧类准确率|RX8 AUROC|
|---|---:|---:|---:|---:|---:|
|fused160::cosine|2.222%|7.278%|2.778%|64.556%|0.5813|
|fused160::mahalanobis_s020|3.378%|2.611%|1.778%|65.972%|0.3896|
|time160::cosine|0.356%|0.389%|2.472%|64.972%|0.5176|
|time160::mahalanobis_s020|0.111%|0.056%|4.667%|62.472%|0.4002|
|frequency160::cosine|0.778%|2.333%|2.944%|64.194%|0.4908|
|frequency160::mahalanobis_s020|0.689%|1.889%|3.444%|64.306%|0.3939|
|pa160::cosine|0.956%|1.278%|0.944%|66.028%|0.3980|
|pa160::mahalanobis_s020|3.367%|1.833%|2.167%|65.250%|0.6199|
|reference29::cosine|2.233%|6.778%|2.167%|64.333%|0.7055|
|reference29::mahalanobis_s020|1.378%|1.389%|1.417%|66.306%|0.6552|
|signature336::cosine|0.700%|0.722%|3.111%|64.889%|0.4289|
|signature336::mahalanobis_s020|1.178%|1.278%|1.944%|66.167%|0.2717|
|spectrum128::cosine|1.844%|6.056%|1.361%|65.222%|0.7493|
|spectrum128::mahalanobis_s020|0.333%|0.056%|4.028%|63.167%|0.2647|
|pa_moments48::cosine|0.656%|2.778%|0.917%|65.694%|0.6677|
|pa_moments48::mahalanobis_s020|1.356%|3.111%|1.556%|65.111%|0.6269|

全V最高未知拒识3.378%，RX8最高7.278%，分别只是本批描述性上界，不是正式方法选择。RX8最高AUROC为0.7493；在当前各类等量的切片中，若要同时达到95% known接受和95% unknown拒识，单调阈值对应ROC至少需要0.9025面积。因此这16条分数不能靠重新选阈值实现目标；不进行target阈值搜索。

## 结论与下一步

这次结果否定了本组四类CE模型上“直接换一个现有物理分支＋中心距离”的简单路线。它不能证明未知识别不可能，也不能证明完整PAMF、不同源表征训练或联合无标签结构利用无效。继续工作优先建立与query互斥的无标签池，再用源代理检验域偏移校正、未知成簇和匿名注册的完整链路；不把本次失败分数扩大到正式目标。

Full eq原始下载已由独立owner完成验证。新Phase3采用RX24-18、D1无标签流与D3独立query的计划保持，三景为low suburban/mid urban/low urban压力分层；数据builder尚在实现，本报告不宣称已建库。既有Phase2暂停任务不恢复。

可核查本地证据：`local_artifacts/pamf_phase3_branch_20261029_r01/{pipeline_state.json,export/smoke.json,export/complete.json,rules/results.json,rules/complete.json,analysis_summary.json}`；完整导出和规则日志同目录logs保留。科学配置见`configs/pamf_phase3_branch_evidence_20261029.json`。
