# F＋跨 RX 同 TX：有界公共相位源检验

状态：PLANNED，尚未运行或取得识别结果。唯一逐行配置见 [experiment.json](experiment.json)。实际代码身份由 Git 提交与 release_commit.txt/resolved_config 保存。

## 唯一假设与矩阵

当前 F＋CROSS_RX 的默认头可能仍依赖接收观测的公共相位。先用冻结模型检验这种识别缺口是否足够大、且在多个源 RX 上重复出现，再决定是否值得另行训练 A_g。相位敏感性不证明相位增强有效，更不证明跨 RX 泛化或联合目标完成。

| 模型 | 数据 | 视图 | 新训练更新 | 新网络包前向 |
|---|---|---|---:|---:|
| 自有 B00／seed2026092701／固定 final20，完整冻结默认头 | 仅训练时见过的 fit L5040，RX3/4/6/8 | identity、+0.37 rad、−0.37 rad | 0 | 15120＋合成 smoke 2 |

该模型来自 `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01` 的 B00。源训练契约 L5040/U45360/V21600，split392005、6TX、day1/2/3；不冒充完整 L6300 或 L5580/Ldev720。自身从零 E20、4440 实际更新，固定 final20、无祖先或目标选模；本次只作冻结推理，不作为新训练父权重。保留三路 SIG_TIME、统一 Sinc、joint160、默认 cosine 头、NoExplicitCFO 和必要内部补偿，既有头未替换、重训或拟合。

旧 carrier/cache 已验证。本次复用原 L6300 缓存，在 IQ 检查与推理前按物理 ID 排除 held RX1，fit 集必须恰好等于该模型训练 L5040；72 个 RX/day/TX 单元各70包。缓存反序列化可能载入其他源角色或旧特征，但不使用它们。held RX1、U、V、target 不作推理或统计，不读取 U TX 真值。

操作仅为 x 乘以常数单位复数，不引入频偏、循环移位、噪声、多径、PA、剪裁或重新归一化。两个固定角度不按源分数搜索，TX/RX 标签不参与变换。正负角度各仅一次，不叠 A_s；数字旋转不声称替换真实 RX。保留以往 CNN 相位训练失败和旧 E200 敏感性结果，其结果不能代替当前 F 的证据。

## 固定判断与停止条件

正负方向分别要求：相对 identity 整体准确率下降至少2个百分点，且至少3/4 fit RX 分别下降至少1个百分点。两个方向都满足才记源敏感性 PASS；否则关闭这个固定有界相位方向，不调角度、换 seed/RX 或选择性救跑。PASS 仅支持另行登记同数据/初始化/前向/监督份额的 scratch E20 B00/B10，不自动训练、扩大种子或接触目标。

输出包含全部15120个默认头决策、logits、总体/RX/TX/day/72单元准确率、macro-F1、混淆矩阵、CE、修复/新增错误、功率误差和实际成本。独立从完整预测与 logits 重算全部决策及216视图单元，不拟合诊断头。相位输入功率相对误差≤1e−6；模型完整 state_dict、eval 状态与梯度保持不变。误差、角色/来源不符、非有限、状态变化、输出冲突或成本不符为技术失败，保留产物，无自动重试。

## 执行、资源与产物

owner `codex/root/F-crossRX-commonphase-20261016`；普通 N607 账号，CPU两线程、CUDA_VISIBLE_DEVICES 为空，不占用 GPU，不修改健康任务。CWD `PROJECT/releases/cvs_F_crossRX_commonphase_sourceL_20261016_r01`，命令 `CVS-RFFI/bin/python -u experiments/cvs_commonphase_source/runner.py`。唯一不可覆盖输出 `PROJECT/runs/20261016-diagnostic-F-crossRX-commonphase-sourceL-s2026092701-m1-r01`。

复用已冻结 backend commit `0ae066adb10a001e834c8c679c5b7b81c20b3ac8` 的真实模型调用路径。单次 N607 preflight、一次发布归档 SHA 比对和远端编译；启动后独立核对 PID/CWD/argv、CPU/CUDA 环境及日志增长。真实匹配 checkpoint 的两个合成包 no-query smoke 是运行第一步，PASS 后直接继续源检验；不增设授权 token。

预期产物：active/submit、真实 checkpoint smoke、resolved_config、fit_L_ids、progress、三视图完整预测、result/completion（或 failure）、完整 stdout、独立复算与图、同 row 报告。IQ、checkpoint、物理 ID、逐包预测/logits、完整日志及运行状态私有；仅代码、配置、登记和汇总报告入 Git。

本次零训练、零头拟合、零教师、零 A_s，15122 实际新增网络包前向；冻结父模型成本 E20/4440更新/4686678网络包前向单列，不重复或隐藏。固定75/25 A_s 的两种子确认已失败并关闭，不以该源相位检验挽救原比例，也不把不同实验的增益相加。

本地10个负例、旋转可逆性/功率和 held RX1 先过滤检查 PASS，未读取真实 IQ 或模型；一次独立 P0/P1 审查 PASS。远端真实 smoke 与实际识别结果以执行产物为准。最终仍须证明 A_g clean 净收益、A_s practical 适应与四因子交互；本次不是正式训练或测试结果。
