# BiSAGE-D92历史D92 E0 Target125实验报告

正式运行报告镜像自`E:\type10-7\automation_reports\CV-SincNet\bisage_d92_hist_e0_target125_20260830_v1\report.md`。

## 预登记摘要

- run ID：`bisage_d92_hist_e0_target125_20260830_v1`
- 状态：`LOCAL_VERIFIED`
- Git提交：`385b0f357440d2869049ecf5293d9782d4b0e3ab`
- 历史配置：5个receiver×5个seed（713102–713106）×5个K/新类切片×3个LEO场景，共125个outer、375个场景单元。
- pilot：`rx_3_19__seed_713102__k_10__new_5`。
- 阶段规则：pilot三个场景的阶段A全部通过才自动进入阶段B；只有pilot通过才发布完整125。
- GPU：pilot固定`cuda:1`。
- query协议：`p2_min_v1`、`VALIDATED_ONCE`、query只读、prediction先完成、独立truth-last scorer后连接truth。
- 四状态：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1`。
- 技术停止仅限协议泄漏、错误矩阵绑定、输出碰撞、启动/闭合/scorer技术失败；低性能不停止。
- 本地验证：30项聚焦测试和`py_compile`通过；独立P0/P1审查的2个P1已定点修复。

最终状态、同row四状态指标、异常、晋级决定和资源证据将在实验完成后追加。

## 技术停止

原run在真实checkpoint无query smoke中发现零方差维度的`SqrtBackward0`产生NaN梯度，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。未打开query、未启动pilot。修复提交为`32998243133e6ec34af263bab2e84435e47b988b`，后续使用不可覆盖新run`bisage_d92_hist_e0_target125_20260830_v1_techfix1`。
