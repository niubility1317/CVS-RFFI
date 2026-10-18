# I/Q 频率相关增益差：固定源统计反证

本次只检验一个新必要假设：当前源 RX 是否具有可跨 TX、日期迁移的低维 I/Q 相对频响增益差。未生成增强 IQ，未启动网络训练，尚无源域或正式性能结果。当前总体模型预算为 E30，本实验模型训练为 0。

物理动机仅限文献确认的频率相关接收 I/Q 失衡现象：[IEEE TSP 2021](https://ieeexplore.ieee.org/abstract/document/9314287)，DOI 10.1109/TSP.2020.3048804。该文不证明当前 ManySig 存在本固定形状、未知 TX 效应可逆或此增强保持 TX 身份。旧静态 RXMIX、共同复 FIR、scalar SRO 和 P50 相对时延路线均保留 FAIL，不换名重跑。

固定观测为每 TX×RX×日期单元的 R(k)=10log10(mean PI(k)/mean PQ(k))，再去除频率均值。只用既有 127 个正 rFFT 频率点，DC/Nyquist 按原统计定义排除；不挑频段或窗。静态 I/Q 增益在这个统计中消去。唯一 RX 形状为 b(k)=(k/127)^2−mean((k/127)^2)，相对 RX1 拟合 gamma_r*b(k)，gamma 在 −1 至 +1 dB 的固定 101 点网格，参考 gamma_1=0。该数值仅为统计拟合范围，不是已获批准的增强幅度。有限窗口、相位混合和真实链路可破坏这种分离，所以必须允许假设失败。

复用 P50 原不可覆盖 run 的 observations.npz：L6300、V27000、6TX×5RX（1/3/4/6/8）×3day，90 格 L70/V300；不新增源 IQ 或模型读取，不访问 U TX、target、query 或 truth。输入复用原 EXACT_MATCH 数据角色；检查完整格数、唯一且不重叠 L/V ID、有限功率及每包交叉谱乘积关系。不新增 hash、数据重验或权限链。

18 个统计检验分别留一个 TX 和一个日期，仅用其余 5TX×其余 2day 的 L 拟合，全部 5 个 RX 仍参与。V 完全只作评价。比较去 RX 均值后的 V 差异与固定相位之外的新增益形状；参照为零 RX 差异、源 L 任意加性 RX 曲线、同 TX 其余 2day 的 RX 曲线。后者只是必要模型比较，未拟合身份分类器。L 每格 70 包按原顺序固定奇偶半样本检验 RX 差异重复性，不假定这些记录 IID。

8 项阈值全部在真实数据运行前固定：V identity SSE>1e−12；L 双半样本 RX 对比功率均值≥0.01 dB² 且每 RX>0；shared≤0.8identity；shared≤0.9additive；≥15/18 检验优于 additive；每 RX 不劣于 additive；shared≤1.05TXconditional；72 个非参考拟合中边界数为 0。任何一项失败就关闭此固定形状，不修改 basis/频段/窗/范围/参考 RX/seed/阈值救援。PASS 也只允许后续独立预登记的有限窗口与默认头身份保持检验，不直接宣称 A_g 或启动 E30。

本地 7 项单元测试已 PASS（0.323 s），包含符号、静态增益消除、边界、不可重复半样本、缺格/非有限/零功率、错误元数据和固定协议；完整 16916400 合成派生实值的源加载及独立复算 PASS，全部 72 解一致。合成绘图在本地被替代，真实远端使用已安装 Matplotlib。8 个源文件 AST、两个生成的远端脚本 AST、10 个成员的模拟发布归档检查 PASS，远端效果均为 mock，不能称真实发布。单次独立 P0/P1 审查待返回。

唯一 run：`20261018-diagnostic-FcrossRX-iqband-LtoV-manysig-m1-r01`；owner：codex/root/iqband-source51；CPU2，CUDA 隐藏，不分配 GPU，不操作其他进程。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_iqband_source_20261018_r01`；固定输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01/observations.npz`；不可覆盖输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-iqband-LtoV-manysig-m1-r01`；日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-diagnostic-FcrossRX-iqband-LtoV-manysig-m1-r01-launch`；本地产物：`E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018`。提交代码后用单包发布、一次 SHA 比较、一次远端编译，launcher 首步 7 单元测试 PASS 即继续。

命令：`CVS-RFFI/bin/python -u -m experiments.cvs_iqband_source51.run`。Windows 发布命令由已验证 Python 执行 `experiments/cvs_iqband_source51/publish.py --stage transfer|launch --commit CODE_COMMIT --source-root WORKTREE --artifact-root LOCAL_ROOT --project-root PROJECT_ROOT --ssh-config E:/type10-7/tools/n607_ssh_config`。实际路径以上述字段为准，提交后补实际 commit。

技术 stop：仅本 run 数据角色/配置错配、非有限、输出碰撞或独立复算失败时保存 failure，不自动重启；不终止其他任务。科学 FAIL 自然完成，写明关闭。产物为实际配置、PID/CWD/cmdline/CUDA 与日志增长证据、主结果、私有拟合数组、独立 CSV/PNG/PDF、完成或失败、真实成本、公开报告/登记/Git。独立复算从原派生 periodograms 重算所有单元与网格解，不复用主拟合数组作为答案；不读 IQ 或模型。

尚无新默认头准确率、clean 代价或跨 seed 收益；本实验不能替代完整 Ag/As 因子实验或最终双提升目标。

单次独立 P0/P1 审查 PASS：未发现直接使真实运行跑错、越权、覆盖或无法产出合法证据的问题；7 项测试与全尺寸合成复算独立执行通过。

实际代码 commit：`0f52826b50ce0ef55ba95971ff4f29da0e689643`。尚无真实源统计结果。

真实实验已完成：技术 VERIFIED，科学 2PASS/6FAIL，6/18 检验改善、66/72 边界；固定路线关闭。无新 IQ/CP/网络/训练/target。完整结果、成本和图见 [final_analysis.md](final_analysis.md)。
