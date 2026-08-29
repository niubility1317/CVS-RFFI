# BiSAGE-D92历史D92 E0 Target125技术修复1实验报告

- run ID：`bisage_d92_hist_e0_target125_20260830_v1_techfix1`
- 状态：`LOCAL_VERIFIED`
- 代码提交：`32998243133e6ec34af263bab2e84435e47b988b`
- 唯一修复：零方差维度在`sqrt`前选择安全方差，保持D92前向等价且保证梯度有限。
- 验证：32项聚焦测试与`py_compile`通过，Git远端OID一致。
- 历史矩阵：5个receiver×5个seed（713102–713106）×5个切片×3个场景，共125个outer、375个场景单元。
- pilot：`rx_3_19__seed_713102__k_10__new_5`；三个场景阶段A全部通过才进入阶段B，pilot通过才进入完整125。
- GPU：`cuda:1`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1`。
- query边界：`p2_min_v1`、`VALIDATED_ONCE`、query只读、prediction完成后独立truth-last评分。
- 四状态：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。
- 技术停止仅限协议泄漏、错误矩阵绑定、输出碰撞、启动/闭合/scorer技术失败；低性能不停止。

最终状态、同row四状态指标、异常、晋级决定和资源证据将在实验完成后追加。
