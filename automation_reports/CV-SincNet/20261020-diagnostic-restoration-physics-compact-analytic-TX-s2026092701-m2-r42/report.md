# R42：不经过F的压缩恢复器TX关联源诊断

状态：**VERIFIED / WAITING_SOURCE**，已发布且正在等待两版完整源预测。24项聚焦检查、CLIhelp、一次独立P0/P1通过。0GPU/神经前向/target；实际源关联结果尚未生成。方法commit `ea8e0ca8b1850e2ca599b7f5c630c91d105918c4`。


状态：VERIFIED / WAITING_SOURCE，已发布且正在等待两版源预测冻结，尚无新诊断结果。此项在 R41 训练结束、源域预测冻结后比较两版恢复 IQ 和原 source-selected R17 的发射机关联，不改变训练或候选排序，不代替 R41 每版七场景正式测试。

固定使用原 R34/R39 的解析坐标、源 L/U 的 8192 条 IQ 校准矩、reference 规则及全部统计，不新拟合特征、分类头或阈值。四组为 mirror64、spectral4、reference24、combined92；保留 spectral 在原 clean 上关联不佳等已有局限，不按结果筛组。校准均值/尺度只由原 L/U 得到，尺度下限0.05；V 不参与任何持久状态更新。

数据为原 R07/R34 相同1800个源V物理样本、clean及六种 practical（郊区/urban各高/中/低仰角）。复用R34 raw/原R17的56个特征单元，从R41两版的冻结源IQ计算56个新单元，共112单元、25200次参数无关变换，0新神经前向/权重加载。R41父版本必须为方法commit `be54678266c95e28bbc18f5a71387c8becf463fa`、两版E20/4940、L126000/U1134000曝光、0TX/F/V/target训练，并具有完整物理参数目标曝光1260000和探针2520000。

控制器首步对原校准和两条实际 L_s IQ 做无query的解析变换检查，随后只等待两版 `source_complete`，不读取源分数、正式分数或正式完成状态。共用R41的score.lock串行执行CPU2线程提取与独立评分，不占GPU，不影响其健康训练/逐行正式测试，也不要求R41等待此项完成。

所有112个特征单元固定且有限、与物理ID对齐后，另起CPU scorer才读源V TX标签。标签只用于冻结后的统计关联，0拟合/训练/部署原型。每TX/RX/day最多2条clean参考，query与参考必须同时不同RX和不同day，保留六TX共同有效条件、空/零参考计数；没有另训TX/RX诊断头。报告同TX与其他TX余弦差、正margin比例（不是分类准确率）、paired clean解析误差及原199次RX/day内query标签打乱对照；打乱范围不改变，描述性对照不称p值或多seed显著性。

完整输出1260个模型/四组/七视图/overall+TX+RX+day记录、252个条件对照和84个模型配对比较。所有模型、视图和组都保留，selected_candidate为null，不设晋级阈值或取消正式测试。比较R41与原R17时与R39口径完全一致，R39作为既有同宽度无物理锚点参照可在同口径下解释，不据这些数值修改已冻结训练、选择父模型或选择性重跑。

准确边界：解析关联可提供独立于F的同TX统计证据，但特征不具有TX专属性，关联提升也不等于每个细节真实；该固定源V研究不能替代未知TX/新payload/真实在轨验证。F50七场景正式识别、波形误差、源物理准确性与同TX关联分别报告，均不单独宣布整体真实性目标完成。

严格0F权重/输出/embedding、0target、0新校准或参数更新；原R17及两版源IQ/数据/日志保持原位。停止条件为父契约/角色/ID错误、缺失/非有限特征、输出碰撞或技术失败，保留产物，不自动重试，不干预健康R41或其他owner。精确矩阵、命令、commit、环境/CWD、独立root与产物在experiment.json中登记；source-only诊断不新增或重跑正式网络版本。


精确执行信息见[登记](experiment.json)。


VERIFIED / WAITING_SOURCE：普通账户新release/归档单SHA/compile已核实，2Ls解析无query smoke PASS，0checkpoint/F/TX/V/target；实际CPU控制器PID/CWD/argv/空CUDA/2线程与等待日志增长已读回。等待R41两版source_complete，不读成绩；尚无解析关联结果。
