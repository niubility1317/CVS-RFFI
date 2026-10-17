# 完整 clean 保留配方的默认 EMA 源域资格诊断

状态：LOCAL_VERIFIED，尚未启动。此诊断不训练、不访问目标或 U IQ。

唯一假设：原默认 EMA 在 clean 上能提供可信身份分布，而同一 V 物理包的 urban 学生存在可教学错误。原门控低接受率不等于 U 标签错误率；U 真值始终隐藏。先检验这一假设，再决定是否实现独立 C4，不直接重放旧 SIG_TIME E50 的 SSL 配方。

固定对照为用户指定完整参考 B01 自身 E20 和 E20+10 两个端点，都必须读取，不能据结果选择父权重。角色精确相同：源 RX1/3/4/6/8、day1/2/3，L6300/U56700隐藏/V27000、split392005。只读同一合法 V1800 及已经存在的 2draw×3scene 完整 practical IQ；无新模拟和测试访问。原 E30 native 收尾有 own-checkpoint 依赖错误，同一冻结 FINAL10（累计30）权重已独立评分恢复；保留事实，不重训、不修改旧产物。

运行前判定：teacher 原 scale30 的默认头 softmax，tau=.95；q=clip((maxp-.95)/.05,0,1)。每个端点均要求 q>0 的 teacher precision≥99%、覆盖≥75%、每个 RX precision≥98%；每个 draw 的 mid/low urban 上 q 加权 teacher正确/student错误比例≥10%。全部满足才是资格 PASS。该诊断不证明训练提升，不用 urban 学生置信度筛监督，不调 threshold、场景、seed 或预算。两端点不是两个独立模型种子。

实现：E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_U_teacher38；主执行脚本为 runner.py，资格计算 maths.py。未来唯一软监督算子 soft_transfer.py 仅实现并验证，不接入训练。它对 teacher stop-gradient，使用 batch mean(q×KL)，不会用入选数重新放大损失；不新增头、特征全对齐或多项一致性。

本地6项检查 PASS＋compile PASS。复用原已验证 F 工厂，仅严格加载旧 student/ownEMA 的900722参数206张量；默认cos6×160/scale30，原前向冻结。launcher 第一步真实 checkpoint 零query smoke，PASS立即继续。每端点14404NN，总28808；0更新/0新teacher训练/0新parent训练/0target。历史父成本20+10=30E/6660只作来源披露。

CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_clean_teacher_source_20261018_r01；环境：N607 ordinary CVS-RFFI Python，CPU2threads，CUDA_VISIBLE_DEVICES空。命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_clean_teacher_source_20261018_r01/experiments/cvs_U_teacher38/runner.py`。唯一路径：/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-clean-teacher-sourceV-e30-s2026092701-m2-r01；日志：/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-clean-teacher-sourceV-e30-s2026092701-m2-r01-launch。Owner：codex/root/Uteacher38。GPU0进程，健康任务不变。技术停止：角色/继承错误、非有限输出、冻结状态变化、输出碰撞或数据越界；保存所有partial，不自动重试。科学失败仍完成两个端点，不扩seed或自动训练。

预期产物：各端点smoke、源预测和complete，整体provenance/source_panel/complete。结果返回后追加同row证据；因0训练且source-only，不启动正式七环境测试。后继训练若开展，仍从零E30并只继承自身固定阶段权重，恢复原13L17U/cos20→resetcos10/逐包nativepractical/4×128完整组织；旧诊断权重不作初始化。
