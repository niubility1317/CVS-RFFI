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
| ECRS-12 | 第13、14、20节 | 实现曲面距离、同TX跨receiver响应预测和不同TX排序损失 | `code/cvsrffi/losses.py` | partial（函数已验证，R6运行接线缺口） | 同TX/异TX合成排序与匹配约束测试；2026-09-05训练装配审计 | 三个loss函数均已实现；但`compute_ecrs_paired_losses()`把`same_tx`计算放在`resp_cls`外层条件内，导致R6虽有`same_tx_cross=true`却实际返回零，R7起才与response CE/diff-TX一并执行 |
| ECRS-13 | 第15、20节 | 实现包内幅度分层50/50 split-fit与clean/LEO双向cross-prediction | `code/cvsrffi/losses.py` | verified（本地） | 分层split、双向误差及配对打乱负对照 | 低置信样本仍参加无标签响应自监督 |
| ECRS-14 | 第17、20、21节 | 保留raw auxiliary CE，增加response CE和gate harm/rescue校准 | `code/cvsrffi/losses.py`、`code/train.py` | verified（本地） | raw分支、response分类、gate界限和rescue/harm测试 | 不允许质量特征直接拼入身份embedding |
| ECRS-15 | 第21、22节 | 按Stage0–Stage6分流梯度和启用损失 | `code/cvsrffi/schedule.py`、`code/train.py` | partial（schedule正确，训练装配未完全服从开关） | 每阶段参数冻结、损失开关测试；R1–R8全路径静态审计 | schedule能产生预期状态；训练装配未单独检查`same_tx_cross`和`gate_calibration`。R6同TX项未实际执行；R7的gate项因`rho=0`成为数值零项；R8首次出现有效rescue/harm时触发AMP/BCELoss异常 |
| ECRS-16 | 第24、31节 | V1使用长度256、固定有效响应维度28、8–12个锚点、64维响应embedding、complex64求解 | `code/model_dual_cvsincnet.py`、`code/train.py` | verified（本地） | 配置解析、8锚点和张量维度测试 | 响应点不足时先降为128点，不删除岭回归 |
| ECRS-17 | 第25节 | 只使用报告给出的初始权重范围与相对尺度岭参数 | `code/train.py`、`code/cvsrffi/schedule.py` | verified（本地） | 参数边界负测与launcher参数解析 | 不写死为最终最优值 |
| ECRS-18 | 第27节 | 记录拟合、可辨识性、泄漏、门控与曲面诊断 | `code/train.py` | verified（R1–R6运行产物） | 每row 98个source-only记录、25088个双视图样本；独立汇总与group-split probe | R1–R6诊断artifact已读回；R7快照时未结束，R8技术中止。逐epoch ECRS子loss虽在runtime计算，但当前CSV/JSONL schema未持久化这些字段 |
| ECRS-19 | 第28、29节 | 实现报告规定的负对照与R0–R11递进矩阵 | `code/tests/`、`code/scripts/` | partial（矩阵声明完整，R6实际执行不独立） | R0–R8 dry-run、Phase2输入拒绝负测、N607运行审计 | R9–R11保持deferred；R6同TX项因训练装配条件实际为零，R7同时新增same-TX、response CE和diff-TX，不能把R6→R7视为单机制递进 |
| ECRS-20 | 第30节 | checkpoint保存basis、`M_ref`、anchor、归一化统计、encoder、gate、response原型/协方差和schema | `code/cvsrffi/checkpoint.py`、`code/train.py` | verified（本地＋R1–R6运行产物） | 真实checkpoint往返；R1–R6最终checkpoint和诊断artifact读回 | schema固定为`ADV3B02:ECRS:z_fused:unit_l2:160:v1` |
| ECRS-21 | 第31节 | 单条LEO IQ独立推理，所有source receiver参与最终训练 | `code/model_dual_cvsincnet.py`、`code/train.py` | verified（本地＋R1–R6 N607） | 无clean输入推理测试；全source receiver训练；每row四场景独立评测 | 不改变Phase2 query边界；当前运行只形成Phase1闭集结果 |
| ECRS-22 | 项目协议4.3节 | 继续使用`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及三段LEO_WEAK日程 | `code/train.py`、正式launcher | verified（R1–R8结构化日志） | CLI解析、E80 CE边界、逐epochCSV/JSONL核对、R1–R6三LEO最终评测 | R7快照尚未完成；R8在E106技术中止，二者不形成最终性能行 |
| ECRS-24 | 用户2026-09-02直接发布指令 | 不运行共享R0，R1–R8分别在GPU0–7从头端到端训练 | `code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh`、launcher测试、新run报告 | verified（N607运行） | direct模式8个候选、0个R0、0个`--init_checkpoint`；GPU映射和独立输出读回 | 用户明确覆盖共享收敛R0前置；属于`USER_OVERRIDE_NON_SHARED_BASELINE`，不得声明严格相邻rung因果增益 |

## 明确不进入V1的设计稿后续项

- 共享低秩可学习基、分块Fisher门控、反事实响应重构/移植、response prototype驱动的Phase2注册，按设计稿第31节在V1证据成立后再进入。
- 上述项目不是被删除，而是保持`deferred`；不得在V1中提前启用，也不得用近似模块替代。

## 设计一致性判定

实现完成前，所有`pending`项必须转为`verified`、`partial`、`deferred`、`rejected`或`blocked`之一。2026-09-05运行审计已将ECRS-12、ECRS-15和ECRS-19降为`partial`：函数和schedule存在，但R6的same-TX loss没有沿实际训练路径生效。因此当前只能声明“主体实现与设计一致，但rung执行接线尚未完全闭合”，不能继续沿用无保留的全量一致结论。

实现的Git目标树固定为`E:\type10-7\github_publish\CVS-RFFI-repo`。本轮隔离工作树为`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-ecrs-v1-parity-fix`，分支为`codex/adv3b02-ecrs-v1-parity-fix-20260901`。`E:\type10-7\code`只作为当前运行副本进行差异核对，不能作为最终编辑或提交位置；同步到运行副本和N607必须发生在Git目标树验证、提交和远端OID读回之后。

## 本地闭合证据

- ECRS聚焦测试：40项通过。
- U_s元数据、clean/LEO配对和启动器补充回归：10项通过。
- 真实ADV3B02基线checkpoint（epoch194）无query smoke：旧权重兼容加载、clean/`leo_clear_weak`配对前向与反向、checkpoint精确往返、单LEO视图推理均通过。
- 原P0/P1定点复审：13项全部`RESOLVED`，总判定`READY`；未增加新gate。
- `git diff --check`通过，仅有行尾转换提示。
- Git Bash宿主通道被错误替换为WSL，`.sh`额外语法检查状态为`FAILED`；启动器的Python解析、R0–R8 dry-run和Phase2输入拒绝负测均已通过。

## N607运行闭合更新（2026-09-05）

- R1–R6：200/200，CSV/JSONL各200条且epoch连续；当前checkpoint与source-val selected-best checkpoint均完成clean和三类LEO评测；诊断artifact存在并已独立解析。
- R7：2026-09-05 23:40快照为194/200，仍在运行；无最终性能结论。
- R8：106/200时发生`torch.nn.functional.binary_cross_entropy`在CUDA AMP下不安全的确定性异常；checkpoint保留，但没有最终clean/LEO结果。该状态是系统性技术失败，不是性能失败。
- R1–R6主结果与完整数据包见`docs/experiments/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_comprehensive_report_20260905.md`及其`docs/experiments/results/.../`目录。
- 未执行Phase2注册、unknown拒识或query truth回流；当前没有候选晋级结论。
