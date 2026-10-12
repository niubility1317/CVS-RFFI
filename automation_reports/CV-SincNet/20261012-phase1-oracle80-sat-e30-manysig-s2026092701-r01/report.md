# 80% 全标签E30＋星地信道增强

用户要求在刚完成的80%全标签训练／20%验证实验上叠加星地信道增强再跑一个。保持相同CVS残差融合、源RX1/3/4/6/8、源day1/2/3、6TX、72000/0/18000角色、sig_i前800/后200划分、seed2026092701、scratch、E30/final30、563步/轮、16890步、AdamW与cosine学习率及FP32。原run不重训、不加载其权重。复用已经VALIDATED_ONCE的数据契约，不因增强改变做数据重验。

唯一研究变化为clean＋LEO弱信道视图及卫星TX交叉熵：cleanCE＋0.68×satCE，无卫星KL、U、EMA或额外辅助损失。采用实际fused clean/satellite拼接前向；两视图共享批统计，辅助CE只用卫星行。增强按batch概率触发，未触发时第二视图为clean复制并保留卫星辅助CE。因此不是单纯变更输入而保持完全相同BN/损失缩放；报告如实披露。模型从零开始，不能把此次预算算作续训。

E1至6使用leo_clear_weak,p=.30；E7至14使用leo_low_elev_weak/leo_rain_weak,p=.60；E15至30三场景并集,p=.80。固定sat_view_seed2027，satelliteCE从E1起计入，避免沿用E80起始导致整个E30没有增强梯度。本次是E30压缩增强安排，不能写成原E200日程严格复现。每轮原始训练样本仍全覆盖72000条，拼接输入行144000条，epoch/update预算相同，计算成本更高并实测报告。

源V只观察健康，固定final30；所有7×168000条clean/六practical预测完整后由独立进程truth-last评分，再与原run固定预测按相同物理ID比较，保存逐场景准确率差及fix/break。测试成绩不回流训练、选模、调参或选择性重跑。已暴露基准，只作单seed探索比较，不声明新的独立盲测。

矩阵、命令、环境/CWD、输入输出/log、GPU/owner和预期artifact见experiment.json。每GPU最多3个总进程、至少12000MB空闲。技术停止规则沿用：角色/预算错误、query越界、非有限训练、启动失败、输出碰撞或无法形成合法prediction时保留产物并停止本run后续阶段；低精度不停止，不干预健康旧任务。launcher第一步自有scratch checkpoint无query smoke，PASS即继续；自动train→freeze→predict→score。

当前状态：本地实现，未发布。执行提交由release_commit.txt与发布证据固定。

本地聚焦验证PASS：真实三种LEO弱信道模拟、E1卫星CE梯度、原始72000样本完整遍历、563批、无U/EMA/辅助网络训练及配置负测。唯一独立P0/P1审查通过，确认native实际启用路径、来源、发布依赖和独立评分衔接。

## 实际发布状态

2026-10-12T21:18:53.553314+08:00：VERIFIED／RUNNING，GPU0。scratchE30、16890步，原80/20物理角色已复用VALIDATED_ONCE，初始checkpoint strict-load/无query smoke通过。实际native参数确认satelliteCE=.68、从E1起启用、fused拼接前向及三段LEO弱信道日程；U/EMA关闭。PID/CWD/argv/GPU和日志增长已独立核实。当前进度：1轮已完成。

执行代码提交`592c5ee67ded914298bb34e0af73c576a4e2025c`。release归档SHA本地/远端比较与compile通过。详细只读证据保留在本地evidence/launch_readback.json。训练后自动固定final30，执行七视图预测→独立truth-last评分→与原run固定预测逐场景比较。尚无新测试结果，不标为实验完成。

后续源域执行核实：已完成E3，concat_sat_active=1.0，该轮实际信道增强batch比例=33.3925%，卫星CE加权均值=0.313802>0，确认增强损失实际参与训练；这不是测试成绩。
