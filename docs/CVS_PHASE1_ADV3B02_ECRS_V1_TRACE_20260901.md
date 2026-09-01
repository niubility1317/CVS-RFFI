# ADV3B02-ECRS-V1设计追溯表

设计来源：`E:\codex\home\attachments\8053de7d-3bda-4fd5-bd38-a5a186bad7d0\pasted-text.txt`

适用范围：本表只追踪设计稿规定的`ADV3B02-ECRS-V1`网络改造、训练接线、checkpoint和验证。未写入设计稿的方法不得加入V1。

| ID | 来源章节 | 设计要求 | 目标文件 | 当前状态 | 验证方式 | 备注 |
|---|---|---|---|---|---|---|
| ECRS-01 | 第1、16、31节 | 保留现有ADV3B02全部主干，在旁路新增局部系统辨识器 | `code/model_dual_cvsincnet.py` | pending | 旧checkpoint严格加载；关闭ECRS时输出逐元素一致 | 禁止替换原PA分支 |
| ECRS-02 | 第7、31节 | `NuisanceEstimator`只估计保守CFO、相位、标量增益 | `code/model_dual_cvsincnet.py` | pending | 合成扰动参数恢复测试 | V1不启用自由RX-IQ纠正和高容量FIR |
| ECRS-03 | 第7、31节 | `AnalyticCanonicalizer`以解析算子执行CFO、相位、增益归一化 | `code/model_dual_cvsincnet.py` | pending | 公共相位/CFO等变性测试 | 不声称恢复真实发射端波形 |
| ECRS-04 | 第8、16节 | 低容量`ContentEstimator`输出规范激励与逐采样置信度 | `code/model_dual_cvsincnet.py` | pending | 形状、有限值、TX梯度隔离测试 | 初期禁止TX分类梯度进入内容估计器 |
| ECRS-05 | 第5、6、31节 | 固定复数相位等变样条响应字典 | `code/model_dual_cvsincnet.py` | pending | 公共相位旋转等变性测试 | V1不启用自由MLP基和可学习低秩基 |
| ECRS-06 | 第9、10、11、23节 | nuisance/fingerprint联合字典、逐采样权重、块状收缩与可微加权岭回归 | `code/model_dual_cvsincnet.py` | pending | Cholesky、10倍岭回退、QR/lstsq回退测试 | 禁止`torch.inverse` |
| ECRS-07 | 第12、16、31节 | 固定锚点采样、协方差传播、64维`z_resp` | `code/model_dual_cvsincnet.py` | pending | 锚点形状、不确定性单调性测试 | 身份比较以曲面值为主，不以自由基原始系数为主 |
| ECRS-08 | 第17、31节 | `rho_max=0.25`受限残差融合到160维身份空间 | `code/model_dual_cvsincnet.py` | pending | `rho∈[0,0.25]`及单位L2范数测试 | gate只读取stop-gradient质量量，不读取类别ID |
| ECRS-09 | 第16、30节 | 模型输出增加raw/response/fused、系数、协方差、质量、锚点、nuisance与内容置信度 | `code/model_dual_cvsincnet.py` | pending | 输出契约测试 | 关闭ECRS时保持旧输出契约 |
| ECRS-10 | 第18、31节 | clean/LEO同步crop、稳定`pair_id`、`physical_sample_id`、`view_type`、`sat_meta` | `code/dataset_wisig.py`、`code/baseline_origin_sat_view.py`、`code/cvsrffi/tensors.py` | pending | clean/LEO元数据与crop一致性测试 | 推理不得依赖clean伴随输入 |
| ECRS-11 | 第15、18、31节 | 保留现有batch级clean+LEO路径；split-fit与pair-cross覆盖`L_s+U_s` | `code/train.py` | pending | 有标签/无标签梯度路由测试 | `U_s`的TX真值继续不可见 |
| ECRS-12 | 第13、14、20节 | 实现曲面距离、同TX跨receiver响应预测和不同TX排序损失 | `code/cvsrffi/losses.py` | pending | 同TX/异TX合成排序测试 | 负样本匹配receiver/day/view/激励覆盖/SNR |
| ECRS-13 | 第15、20节 | 实现包内幅度分层50/50 split-fit与clean/LEO双向cross-prediction | `code/cvsrffi/losses.py` | pending | 配对打乱负对照 | 低置信样本仍参加无标签响应自监督 |
| ECRS-14 | 第17、20、21节 | 保留raw auxiliary CE，增加response CE和gate harm/rescue校准 | `code/cvsrffi/losses.py`、`code/train.py` | pending | raw分支不退化与gate界限测试 | 不允许质量特征直接拼入身份embedding |
| ECRS-15 | 第21、22节 | 按Stage0–Stage6分流梯度和启用损失 | `code/cvsrffi/schedule.py`、`code/train.py` | pending | 每阶段参数冻结与损失开关测试 | V1正式路线止于固定基；学习基仅保留后续入口且默认关闭 |
| ECRS-16 | 第24、31节 | V1使用长度256、固定有效响应维度28、8–12个锚点、64维响应embedding、complex64求解 | `code/model_dual_cvsincnet.py`、`code/train.py` | pending | 配置解析和张量维度测试 | 响应点不足时先降为128点，不删除岭回归 |
| ECRS-17 | 第25节 | 只使用报告给出的初始权重范围与相对尺度岭参数 | `code/train.py`、`code/cvsrffi/schedule.py` | pending | 参数边界负测 | 不写死为最终最优值 |
| ECRS-18 | 第27节 | 记录拟合、可辨识性、泄漏、门控与曲面诊断 | `code/train.py` | pending | 日志字段完整性测试 | TX/RX probe为训练独立诊断 |
| ECRS-19 | 第28、29节 | 实现报告规定的负对照与R0–R11递进矩阵 | `code/tests/`、`code/scripts/` | pending | launcher dry-run与结果行绑定 | V1先执行R0–R8；R9–R11须在V1证据满足后启用 |
| ECRS-20 | 第30节 | checkpoint保存basis、`M_ref`、anchor、归一化统计、encoder、gate、response原型/协方差和schema | `code/cvsrffi/checkpoint.py`、`code/train.py` | pending | 保存—加载—单视图推理往返测试 | schema固定为`ADV3B02:ECRS:z_fused:unit_l2:160:v1` |
| ECRS-21 | 第31节 | 单条LEO IQ独立推理，所有source receiver参与最终训练 | `code/model_dual_cvsincnet.py`、`code/train.py` | pending | 无clean输入推理测试；receiver覆盖统计 | 不改变Phase2 query边界 |
| ECRS-22 | 项目协议4.3节 | 继续使用`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及三段LEO_WEAK日程 | `code/train.py`、正式launcher | pending | CLI解析与dry-run | 最终分别报告clean和三个`leo_*_weak`场景 |

## 明确不进入V1的设计稿后续项

- 共享低秩可学习基、分块Fisher门控、反事实响应重构/移植、response prototype驱动的Phase2注册，按设计稿第31节在V1证据成立后再进入。
- 上述项目不是被删除，而是保持`deferred`；不得在V1中提前启用，也不得用近似模块替代。

## 设计一致性判定

实现完成前，所有`pending`项必须转为`verified`、`deferred`、`rejected`或`blocked`之一。只有ECRS-01至ECRS-22全部有证据，且V1范围内不存在`pending`，才能声明“与设计稿一致”。
