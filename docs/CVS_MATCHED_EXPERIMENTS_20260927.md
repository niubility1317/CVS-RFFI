# CVS与D92固定版本实验

本次按用户确认运行原始DAOT A1＋FastTrust-RC4 residual_noeq，后续接D92 E0真实256维v2。Phase1训练代码固定于`9b1ddd939fcabc74c77c94a62fef8e5aad07c0e9`，N607 release为`cvs_rc4_matched_20260927_r01`。

## Phase1

- Run：`20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01`。
- 模型seed：392005、2026092701、2026092702、2026092703、2026092704；历史392005与其余四seed分开汇总。
- 6300个L、56700个U、27000个V，与外部对比的物理角色逐项一致。从零训练，EMA来自本次student，不加载历史checkpoint。
- 200轮后使用`final_ssdg.pth`，不选最佳轮。原始算法与信道训练参数保留，关闭周期目标评分和final weak reference。
- `residual_noeq`为residual/post_sync/无额外均衡；receiver seed2027。CVS原生训练随机视图依赖模型seed、epoch及batch，不声称与对比方法视图逐样本相同。
- CVS用U loader定义每轮，不能将相同200轮称作相同计算预算。记录实际步数、资源及耗时。

入口：`experiments/adv3b02_xuc/tools/publish_rc4_matched.py`。实体配置位于`experiments/adv3b02_xuc/configs/matched_20260927/`，首次运行执行真实checkpoint无query smoke。发布器拒绝覆盖已有release/run；恢复时先读记录与实时进程，不能重复调用来“续跑”。

## 最终测试与D92

Run：`20260927-phase2-cvs-d92-practical-manytx-m5-r01`。入口为`tools/publish_cvs_matched_pipeline.py`，唯一配置为`configs/cvs_d92_matched_20260927.json`。

每个CPU lane等待对应Phase1完成并核对checkpoint完整来源与契约。先用本次最终编码器提取本次全部L特征，构建rank-3低秩、INT8＋FP16尺度、4096-bin P90半径的v2聚合组件。Phase2不读源域单样本，只消费此组件和冻结编码器特征。

D92固定`P2-256-FULL`：identity160＋FFT96，块权重4、块归一化后整体归一化、D89半径可靠性谱、D81 Cauchy中心、D92旧／新任务等权自动收缩协方差、support内LOO和F3双层INT8。不是含RF32的288维`P2-FULL`；内部API使用288列输入容器，实际注册状态强制检查为256维。

共享矩阵包含新类数0、2、5、10、20，K为1、5、10、20，7个RX、3个星地场景和5个support seed，共2100个划分／模型seed。无新类分支只使用原执行器已有的旧类输入装配；新2类只扩展原协方差实现的允许注册类数，其数学计算不变。D92是support驱动的注册状态拟合，不更新编码器梯度。

Phase1最终测试复用对比实验已有clean／星地固定capsule；Phase2复用`residual-noeq-ba667eee4fb061055e4c08b5`，不重采信道或重划support/query。每条query只读、面对全部注册类别，完整prediction后才由独立scorer接入truth。5个lane完成前不评分，不把目标结果回流选模或选择性重跑。

输出包含每row的完整启动参数、resolved配置、来源检查、source特征准备、v2组件、预测与阶段日志；run根目录保存`state.json`、`phase1_final_results.json`和`scored_results.json`。前两阶段运行状态与最终评分状态分开记录；等待源模型不等于D92已经完成。
