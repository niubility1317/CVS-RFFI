# PAMF未知设备跨RX身份结构冻结诊断

## 预登记

Run `20261030-phase3-star-open-feature-diagnostic-s2026092701-r01`，执行代码 `55977eec2cf0c2bd09d7c6856179f919e9f0923d`。原F0 fused160和原r02 S0/S1/M各fused160/open64共七种表征全部固定比较；不训练、不换主M、不挑权重/阈值。完整方法/矩阵/边界见`docs/PAMF_PHASE3_OPEN_FEATURE_DIAGNOSTIC_20261030.md`，行矩阵唯一维护experiment.json。

本run只读既有source novel3150条G10P1（19-15/19-16/7-1各1050、source5RX/3day），parent原F0E50/11100及source-onlyr02三arm2220/cumulative13320/code1029bc4...，与cache原F0/DURA祖先一致。源RX已见、源novel多次开发暴露，不是新RX/新TX盲测。L/V/OE/Us/target U/Q/正式RX24-18新IQ0，G加载/前向0；全postPhase1新IQ仍星地，同P1不重抽频道。

四实际checkpoint两条合成G10 smoke和frozenstate检查先于source cache打开；同次前向unit fused160/open64，真zero保留。输出4NPZ仅IDs/RX/day/features，无TX/K；全部七表征完整、所有四条件1NN索引都先于唯一source truth读取。CPU FP64 blocked128、候选排除自身、按opaqueID稳定ties；RX/day可过滤，TX/K不能影响NN选择。独立truth端只判sameTX、宏均值及RX/day/TX、余弦类内外gap、谱/有效秩/参与秩与TX/RX/day组间方差比；缺候选N/A、常量秩0，不做物理解耦因果证明。

这些是同池参考检索/几何诊断，不是无标签分类或未来匿名注册。AN/HOW/UDA/REGN/A、eligiblefalse/DIAGNOSTIC_ONLY，完整targets v3 CK90/RU90/F0loss2pp及匿名未来目标保持ACTIVE。本run不通过source统计直接晋级或触碰formal。

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-open-feature-diagnostic-s2026092701-r01-release`；owner `codex-phase3-open-feature-diagnostic-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_open_feature_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-open-feature-diagnostic-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-open-feature-diagnostic-s2026092701-r01`；实际GPUfree≥4096MB时只新增一个export worker，允许foreign共卡、每卡≤2新增own；完成自动接独立CUDAempty/BLAS1CPU scorer。其他用户停止/修改/重启0。
- 新增feature-forward12600实物+8合成，G0/updates0；只有3150既有物理记录。3完整nativeparent容器+F0约3090682驻留参数，仅作私有诊断，不称部署或压缩。4NPZ/日志不Git。
- 聚焦export3 PASS2.75s、launcher3 PASS2.80s、scorer5 PASS9sub0.95s；不是一次合并suite。6AST/实际runtime接口PASS；唯一直接P0/P1 PASS。一次preflight只查父模型/数据句柄/文件存在/实际显存，未读IQ/truth值；归档一次比较/compile、首步actualcheckpoint smoke、独立PID/CWD/argv/GPU/log增长、全部feature/邻居固定后truth-last，不增hash/签章/批准链。
- 预期launch/pipeline/config/logs；features F0/S0/S1/M NPZ、smoke/complete；score results/complete；observations。非法祖先/角色、非有限、输出碰撞/不可预测技术错误保留，不自动重试；UNKNOWN先核实原PID/root，不停止foreign。当前尚未宣称已启动或得到诊断结果。


## 实际完成与发现

VERIFIED frozen source-novel3150/sevenrepr/all4NNchoices before sole source truth. F0 fused crossRXday76.603175%, M fused76.825397%, M open76.984127%, S1 open77.587302%. M open RXvariance16.286785% vs F0 2.887514%, not wholesale collapse/no causalproof. Target/G/updates0, AN/REG N/A, fullgoalACTIVE.

详见`docs/PAMF_PHASE3_OPEN_FEATURE_GEOMETRY_RESULTS_20261030.md`。
