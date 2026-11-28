# LoRa 与 WiSig 共用源域冻结教师指导门控筛查

- run_id：`20261128-phase1-shared-teacher-selector-s2026112937-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两数据集源域教师指导训练正在进行。

## 目的与对照

用户授权的共同识别器指导训练：冻结各数据集默认识别器，用源 U 同包 clean 教师分布与两专家恢复分布的 KL 偏好训练共同选择器。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码与授权协议提交 `7a263539a8366f62699dca566b566d4d9e976da2`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3767727、LoRa PID 3767738 / GPU 0、WiSig PID 3767739 / GPU 5。独立核查 CWD、命令、显存、真实专家/教师首步及非零有限梯度，日志持续增长。发布前 GPU 0、5 可用显存约 19.9、16.6 GiB，保留其他 owner。一次独立 P0/P1 准备审查通过；合成张量和共享网络测试通过。

## 结果与覆盖

尚无筛查或正式结果。用户明确授权当前两数据集源域教师指导训练，详细权限和祖先见项目协议及 `docs/goals/SHARED_DURA_REALGAIN_TEACHER_PROPOSAL_20261128.md`。共用门控以 LoRa W24 E104 或 WiSig Tri-SIG-TIME SSL F0 默认头在源 U clean/模拟受扰上的分布生成 KL 专家偏好伪标签；教师及专家冻结，不读 U TX、V、目标或目标统计。各侧 1,000 步，LoRa/WiSig 每步 batch 4/24、教师样本前向 24,000/144,000，源 V 多次曝光。原教师训练成本、LoRa 实增益教师和伪标签祖先必须单列。训练完成后按预登记自动冻结 clean、低城区同包预测并独立 CPU truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
