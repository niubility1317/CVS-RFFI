# H6部署化与HardPair实验报告

正式控制面报告：`E:\type10-7\automation_reports\CV-SincNet\stage2_sf_tapft_h6_deploy_hardpair_s392002_20260828_r1\report.md`。

当前状态：`LOCAL_VERIFIED`。

- 已实现固定full-support deployment-only入口、FP32/FP16冻结前缀缓存、严格delta和support-derived HardPair。
- 聚焦回归88项通过；真实ADV3B02 checkpoint和60条support无query smoke通过。
- FP32缓存对完整路径的最大logit差和许可norm梯度差均为0，prediction完全一致；FP16输出有限。
- 独立P0/P1审查发现并修复真实CVS head需要`return_emb=True`的问题，定点复审PASS。

本镜像将在N607运行、truth-last评分和最终分析后补充完整数据。
