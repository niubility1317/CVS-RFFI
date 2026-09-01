# ADV3B02-ECRS-V1设计追溯表

设计来源：`E:\codex\home\attachments\8053de7d-3bda-4fd5-bd38-a5a186bad7d0\pasted-text.txt`

适用范围：本表只追踪设计稿规定的`ADV3B02-ECRS-V1`网络改造、训练接线、checkpoint和验证。未写入设计稿的方法不得加入V1。

| ID | 来源章节 | 设计要求 | 目标文件 | 当前状态 | 验证方式 | 备注 |
|---|---|---|---|---|---|---|
| ECRS-01 | 第1、16、31节 | 保留现有ADV3B02全部主干，在旁路新增局部系统辨识器 | `code/model_dual_cvsincnet.py` | verified（本地） | 旧checkpoint兼容加载；关闭ECRS时输出逐元素一致 | 禁止替换原PA分支；保留现有SAT-Anchor、CRRA及容量对照接口，不将其擅自并入ECRS |
| ECRS-02 | 第7、31节 | `NuisanceEstimator`只估计保守CFO、相位、标量增益 | `code/model_dual_cvsincnet.py` | verified（本地） | 合成扰动参数恢复测试 | V1不启用自由RX-IQ纠正和高容量FIR |
| ECRS-03 | 第7、31节 | `AnalyticCanonicalizer`以解析算子执行CFO、相位、增益归一化 | `code/model_dual_cvsincnet.py` | verified（本地） | 公共相位/CFO等变性与cycle测试 | 不声称恢复真实发射端波形 |
| ECRS-04 | 第8、16节 | 低容量`ContentEstimator`输出规范激励与逐采样置信度 | `code/model_dual_cvsincnet.py` | verified（本地） | 形状、masked reconstruction、TX梯度隔离测试 | 初期禁止TX分类梯度进入内容估计器 |
| ECRS-05 | 第5、6、31节 | 固定复数相位等变样条响应字典 | `code/model_dual_cvsincnet.py` | verified（本地） | 公共相位旋转等变性及fixed MP 1/3/5阶测试 | V1不启用自由MLP基和可学习低秩基 |
| ECRS-06 | 第9、10、11、23节 | nuisance/fingerprint联合字典、逐采样权重、块状收缩与可微加权岭回归 | `code/model_dual_cvsincnet.py` | verified（本地） | 联合回归、Cholesky、10倍岭回退、QR/lstsq回退测试 | 禁止`torch.inverse` |
| ECRS-07 | 第12、16、31节 | 固定锚点采样、协方差传播、64维`z_resp` | `code/model_dual_cvsincnet.py` | verified（本地） | 8锚点、可训练16→64编码器、不确定性传播测试 | 身份比较以曲面值为主，不以自由基原始系数为主 |
| ECRS-08 | 第17、31节 | `rho_max=0.25`受限残差融合到160维身份空间 | `code/model_dual_cvsincnet.py` | verified（本地） | `rho∈[0,0.25]`及单位L2范数测试 | gate只读取stop-gradient质量量，不读取类别ID |
| ECRS-09 | 第16、30节 | 模型输出增加raw/response/fused、系数、协方差、质量、锚点、nuisance与内容置信度 | `code/model_dual_cvsincnet.py` | verified（本地） | 输出契约测试 | 关闭ECRS时保持旧输出契约 |
| ECRS-10 | 第18、31节 | clean/LEO同步crop、稳定`pair_id`、`physical_sample_id`、`view_type`、`sat_meta` | `code/dataset_wisig.py`、`code/baseline_origin_sat_view.py`、`code/cvsrffi/tensors.py` | verified（本地） | clean/LEO元数据、crop一致性及U_s无TX真值测试 | 推理不得依赖clean伴随输入 |
| ECRS-11 | 第15、18、31节 | 保留现有batch级clean+LEO路径；split-fit与pair-cross覆盖`L_s+U_s` | `code/train.py` | verified（本地） | 有标签/无标签梯度路由测试 | `U_s`的TX真值继续不可见 |
| ECRS-12 | 第13、14、20节 | 实现曲面距离、同TX跨receiver响应预测和不同TX排序损失 | `code/cvsrffi/losses.py` | verified（本地） | 同TX/异TX合成排序与匹配约束测试 | 负样本匹配receiver/day/view/激励覆盖/SNR |
| ECRS-13 | 第15、20节 | 实现包内幅度分层50/50 split-fit与clean/LEO双向cross-prediction | `code/cvsrffi/losses.py` | verified（本地） | 分层split、双向误差及配对打乱负对照 | 低置信样本仍参加无标签响应自监督 |
| ECRS-14 | 第17、20、21节 | 保留raw auxiliary CE，增加response CE和gate harm/rescue校准 | `code/cvsrffi/losses.py`、`code/train.py` | verified（本地） | raw分支、response分类、gate界限和rescue/harm测试 | 不允许质量特征直接拼入身份embedding |
| ECRS-15 | 第21、22节 | 按Stage0–Stage6分流梯度和启用损失 | `code/cvsrffi/schedule.py`、`code/train.py` | verified（本地） | 每阶段参数冻结、损失开关及R5分块收缩启用测试 | V1正式路线止于固定基；学习基仅保留后续入口且默认关闭 |
| ECRS-16 | 第24、31节 | V1使用长度256、固定有效响应维度28、8–12个锚点、64维响应embedding、complex64求解 | `code/model_dual_cvsincnet.py`、`code/train.py` | verified（本地） | 配置解析、8锚点和张量维度测试 | 响应点不足时先降为128点，不删除岭回归 |
| ECRS-17 | 第25节 | 只使用报告给出的初始权重范围与相对尺度岭参数 | `code/train.py`、`code/cvsrffi/schedule.py` | verified（本地） | 参数边界负测与launcher参数解析 | 不写死为最终最优值 |
| ECRS-18 | 第27节 | 记录拟合、可辨识性、泄漏、门控与曲面诊断 | `code/train.py` | verified（本地） | 日志字段、独立probe payload和不可覆盖artifact测试 | TX/RX probe为训练独立诊断 |
| ECRS-19 | 第28、29节 | 实现报告规定的负对照与R0–R11递进矩阵 | `code/tests/`、`code/scripts/` | verified（V1范围） | R0–R8 dry-run、Phase2输入拒绝负测与结果行绑定 | R9–R11按设计稿保持deferred，须在V1证据满足后启用 |
| ECRS-20 | 第30节 | checkpoint保存basis、`M_ref`、anchor、归一化统计、encoder、gate、response原型/协方差和schema | `code/cvsrffi/checkpoint.py`、`code/train.py` | verified（本地） | 真实ADV3B02 checkpoint保存—加载—单视图推理精确往返 | schema固定为`ADV3B02:ECRS:z_fused:unit_l2:160:v1` |
| ECRS-21 | 第31节 | 单条LEO IQ独立推理，所有source receiver参与最终训练 | `code/model_dual_cvsincnet.py`、`code/train.py` | verified（本地实现） | 无clean输入推理测试；launcher保持全部source receiver | 不改变Phase2 query边界；尚未启动N607性能实验 |
| ECRS-22 | 项目协议4.3节 | 继续使用`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及三段LEO_WEAK日程 | `code/train.py`、正式launcher | verified（本地实现） | CLI解析、E80 CE边界与R0–R8 dry-run | 最终clean和三个`leo_*_weak`实测须由后续N607实验产生 |

## 明确不进入V1的设计稿后续项

- 共享低秩可学习基、分块Fisher门控、反事实响应重构/移植、response prototype驱动的Phase2注册，按设计稿第31节在V1证据成立后再进入。
- 上述项目不是被删除，而是保持`deferred`；不得在V1中提前启用，也不得用近似模块替代。

## 设计一致性判定

实现完成前，所有`pending`项必须转为`verified`、`deferred`、`rejected`或`blocked`之一。只有ECRS-01至ECRS-22全部有证据，且V1范围内不存在`pending`，才能声明“与设计稿一致”。

实现的Git目标树固定为`E:\type10-7\github_publish\CVS-RFFI-repo`。本轮隔离工作树为`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-ecrs-v1-parity-fix`，分支为`codex/adv3b02-ecrs-v1-parity-fix-20260901`。`E:\type10-7\code`只作为当前运行副本进行差异核对，不能作为最终编辑或提交位置；同步到运行副本和N607必须发生在Git目标树验证、提交和远端OID读回之后。

## 本地闭合证据

- ECRS聚焦测试：40项通过。
- U_s元数据、clean/LEO配对和启动器补充回归：10项通过。
- 真实ADV3B02基线checkpoint（epoch194）无query smoke：旧权重兼容加载、clean/`leo_clear_weak`配对前向与反向、checkpoint精确往返、单LEO视图推理均通过。
- 原P0/P1定点复审：13项全部`RESOLVED`，总判定`READY`；未增加新gate。
- `git diff --check`通过，仅有行尾转换提示。
- Git Bash宿主通道被错误替换为WSL，`.sh`额外语法检查状态为`FAILED`；启动器的Python解析、R0–R8 dry-run和Phase2输入拒绝负测均已通过。

以上只证明本地实现与报告V1设计闭合，不等同于N607实验已经启动、性能指标已经获得或候选已经晋级。
