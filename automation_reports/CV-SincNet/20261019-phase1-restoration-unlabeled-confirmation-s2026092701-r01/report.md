# 恢复器自监督校核与独立评分恢复

状态：ANALYZED / VERIFIED，新增训练与 GPU 预测均为0。父实验：`20261019-phase1-pamf-restoration-selfsup-manysig-m7-r01`；校核代码：`c9aaaec09b66da475fe5b6322b14847994d40d96`。

恢复器训练使用源域无 TX 标签 IQ，经已知星地模拟器生成退化—原观测配对；这是合成退化自监督，允许使用原始源域 clean 观测。原观测仍含地面接收机、传播与噪声，不能称为理想发射波形。L_s 与 U_s 的身份标签均不进入恢复训练，也没有伪标签、分类交叉熵或冻结 F 的训练梯度。已有 F 的预训练使用过标签，整个识别系统不能称为全无监督。

原七网络比较的 source V 选模使用 TX 标签。这里增加一个只接收波形指标的独立校核：clean→clean NMSE≤0.02，然后最小化六 practical 平均原坐标 IQ NMSE；禁止输入 TX、F 预测、F 特征或分类准确率。该校核是在源域七行诊断完成之后补充，不能追溯声称为事前盲选。校核选择 U-Net，与原选择相同，因此直接复用原先冻结的全部预测，增加训练与 GPU 预测次数均为 0。

六项窄检查通过：恢复目标无分类/教师依赖；扩散 v 目标和完整 DDIM 的解析端点；流匹配完整 Euler 端点；真实推理入口不调用依赖 clean 的训练路径；无 TX 选模拒绝监督字段；真值路径唯一性与不提前读文件。端点检查验证公式实现，不保证网络已经学到输入对应的身份细节。

原目标预测：1,200 个固定均匀物理 ID，clean＋六 practical 全部完成，`TX_truth_reads=0`。原评分失败点是硬编码本地 `scoring/truth.json` 目录，而 N607 已有数据使用 `data/truth.json`；在成功读真值前报错。本恢复仅用新不可覆盖 run 目录引用原 prediction、原 target_inputs 和原 final checkpoint，调用已有独立 CPU scorer。原 run、失败日志及预测保持。

N607 已有 Python 环境，CWD：`releases/pamf_restoration_selfsup_20261019_r01`。仅 CPU，无 GPU worker。精确调用见 experiment.json。任一完整预测/物理 ID 检查失败就停止，不自动训练或重预测。预期产物：独立 metrics.json、launch.json、完整 score.log、completion.json。完成标准是七视图独立真值评分与同 row 解释。

选模结果见 [unlabeled_decision.json](results/unlabeled_decision.json)。目标数据是固定公共接收机会话下新加 practical 的 Phase1 配对诊断，不是历史卫星观测或 Phase2 结果。单模型 seed、1,200 条诊断不代表完整矩阵或部署证明。

独立读回：原预测7视图×1200覆盖匹配，CPU scorer完成并退出，metrics状态SCORED；原失败日志保留。主 AsF low urban49.83%→48.50%，clean84.17%→83.33%；四个其他practical场景局部提高，科学身份标准仍未通过。完整结果见 [target_confirmation.json](results/target_confirmation.json)，总体分析与23/6/1报告核对见父实验report。
