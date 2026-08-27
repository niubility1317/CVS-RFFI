# SF-TAPFT Fast-Strong V3实验报告

## 预登记

- 状态：`LOCAL_VERIFIED`。
- run ID：`stage2_sf_tapft_fast_strong_v3_s392002_20260827_r1`。
- Git提交：`5ec7f1fac38ef674e613f2e07a49d7338715f3a8`。
- 科学边界：`p2_min_v1`、`VALIDATED_ONCE`，复用capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`和split=`stage2b-rx20-1-seed713101-before-support-prefix`；旧6类，每类K=10，共60条独立物理support；不注册新类。
- 本地环境/CWD：Conda`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`。
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet`。
- 矩阵：`configs/stage2_sf_tapft_fast_strong_v3_s392002_20260827.json`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入support：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2b_sclba_a_t5t25_s713101_20260824_v1/input/support_rx20_1_k10_clear_smoke.npz`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_fast_strong_v3_s392002_20260827_r1`，每行使用不可覆盖子目录。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_fast_strong_v3_s392002_20260827_r1`。
- GPU分配：GPU0运行H0/H1，GPU1运行H2/H3，GPU2运行H4/H5，GPU3运行H6；任何GPU均不超过两个本run训练任务，实际启动前以资源盘点为准。
- 启动命令模板：`/usr/bin/time -v <python> code/scripts/run_sf_tapft_slim_matrix_row.py --matrix <matrix> --row-id <H0-H6> --output-dir <run-root>/<row>/output --device cuda:<gpu> --folds 4`。
- 直接技术停止规则：仅在协议/query泄漏、错误checkout或输入、输出碰撞、无prediction闭合、launcher级故障，或至少两行出现相同确定性pre-prediction异常时，定点停止本run所属进程树。低性能不停止。
- 预期artifact：每行`selection.json`、`sf_tapft_clean_single_bundle.pt`、`sf_tapft_delta_bundle.pt`、stdout/stderr、GNU time和GPU采样；prediction闭合后再由独立scorer连接truth。

## 因果矩阵

|行|相对上一行的唯一主要变化|总优化步|训练主干forward上限|
|---|---|---:|---:|
|H0|S15轨迹，300步/ref4500|300|300|
|H1|追加150步局部余弦尾段|450|450|
|H2|将尾段由150扩至300步|600|600|
|H3|H1后追加70步缓存head-only精修|520|450|
|H4|H3加入许可delta EMA，beta=0.99|520|450|
|H5|H4加入class-adaptive rho与可靠性head anchor|520|450|
|H6|S02的t3-only Norm结构叠加H3流程|520|450|

所有行均保持balanced CE、label smoothing=0.05、LOO-proto权重0.5、L2-SP=1e-4、scale=8、KD=0和support OOF温度拟合。OOF温度不改变类别argmax；欠拟合行仍必须同时通过BA与floor，不能仅凭NLL晋级。

## 预登记判断规则

- support OOF门槛：BA≥86.17%、最低类别召回≥60%、NLL≤0.5394。
- 相对历史强工作点的实质提升：BA>86.67%，或NLL<0.5094且floor不降低。
- 资源目标：变化元素约≤1500、无Adapter/完整block、delta bundle<10KB、Query每条仍单独判别且推理计算不增加。
- support完成后选择满足门槛的最小候选；所有H0–H6仍保留同row结果，不删除负结果。

## 最大真实Query闭合预登记

训练、OOF选择、full-support refit和温度拟合全部冻结后，才打开既有两组真实query预测输入：原Q60与独立Q120。两组物理ID零重叠，合并为Q180，每类30条。预测阶段不读取query truth、角色或类别计数；两组prediction均完整后，独立scorer最后连接truth。报告采用`DA0_REG0`与`DA1_REG0`，逐类给出30条分母下的准确率、总体BA、class floor、NLL、ECE、正确数变化和成对翻转。`REG0`不定义新类准确率，记为`N/A`。

## 本地验证

- Fast-Strong矩阵7/7行成功解析为`SFTAPFTConfig`。
- 聚焦测试：81项通过；唯一警告为旧Torch AMP命名空间的弃用提示，不影响运行。
- 独立P0/P1审查曾发现函数插入位置错误；定点修复后复审PASS，selection与runner闭合恢复。
- Git远端核对：`origin/codex/meta-adapter-tri-r4-v1-20260824`OID等于本地`5ec7f1fac38ef674e613f2e07a49d7338715f3a8`。

## 发布与结果

待N607预检、release同步、真实启动与Q180闭合后追加。
