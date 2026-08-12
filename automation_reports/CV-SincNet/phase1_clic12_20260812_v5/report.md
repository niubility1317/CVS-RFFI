# Phase1 CLIC 12臂训练v5预注册与运行报告

## 1. 状态与目标

- 实验ID：`phase1_clic12_20260812_v5`
- 当前状态：`PREREGISTERED / NOT_LAUNCHED`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 目标：完成F1—F6×C/G共12臂、每臂40epoch的P1-CLIC源域训练；训练技术闭合后，对目标域统一叠加`LEO weak`星地信道，联合评估域泛化和未知类拒识。
- v1—v4全部是封存的系统性技术失败，均为`NO_PERFORMANCE_RESULT`；v5使用新的run/log/output路径，不恢复、覆盖或重试任何旧run。

## 2. v5启动依据

- 核心修复commit：`f43f313e`（`fix: bind CLIC trainer batch metadata`）。真实`move_batch`输出为`extra=(domain,metadata)`；修复后从metadata严格提取128行`base_index/sig_i/local_label`，并拒绝bool、浮点、字符串、非有限值、负索引及越界标签。
- 本地`ssr-gpu`验证：`py_compile`通过；TX分区与CLIC联合回归195项通过；独立复审`P0=0/P1=0`。
- N607同版本真实入口烟测`phase1_clic_smoke3_20260812_v1`已通过：F1C/F1G分别完成3个optimizer batch，clear/low_elev/rain各1；AMP=3、graph release=3、proxy/query/target/selection=0；两进程exit0，无checkpoint/terminal/prediction/score及性能读取。
- 烟测报告：[report.md](../phase1_clic_smoke3_20260812_v1/report.md)，commit=`67e9dde4`，SHA256=`21D05D8FC16387597CC9D60788E29801C27963AA75359C45A56D7846A742F85B`。

## 3. 冻结矩阵与方法

- 矩阵：F1—F6×C/G=12臂；C=`raw_phase_control`，G=`complex_local_invariant_curvature`。
- G仅从每个样本同一份received_i提取固定lag=`{1,2,4,8}`的多尺度三点复曲率token；不增加观测、不融合多场景，C/G除operator外完全同配置。
- seed=`7281164`；40epoch；batch=128；AdamW；lr=`2e-4`；共同L_base=`clean CE+0.10×KL(clean-stopgrad→single-LEO)`；LEO训练场景固定轮转`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；checkpoint=`final_only`。
- 每折TX角色为4个source-L训练TX、1个known-validation TX、1个proxy-unknown TX，三者互斥；proxy仅用于后冻结连续未知分数诊断，训练fit/forward/update为0。
- target/query/truth/role训练访问为0；不按目标结果调参、选场景、选seed、早停或重试。
- GPU映射：0=`F1C,F5G`；1=`F1G,F5C`；2=`F2C,F6G`；3=`F2G,F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。每张GPU最多2个训练进程，符合默认资源上限。

## 4. 发布与启动合同

- 源版本：Git commit `f43f313e`的干净archive；Task7 dirty/untracked文件不得进入release。
- 已验证archive：SHA256=`4F96E4203830809BA807750F03921EE21F26D79EDEE5C401BD179D5C87B3A03F`，bytes=`266874880`。
- 复用已原子落地且逐文件验签的只读release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_smoke3_20260812_v1_f43f313e`；本run不重复SCP/解包，不重新做已闭合的数据或配置验证，只复核release hash、GPU资源和新run路径。
- launcher：上述release内`code/scripts/launch_phase1_clic12_20260811.sh`；环境覆盖`RUN_ID=phase1_clic12_20260812_v5`、`CODE_ROOT=<release>/code`。
- run/log/outer：`runs/phase1_clic12_20260812_v5`、`logs/phase1_clic12_20260812_v5`、项目根`phase1_clic12_20260812_v5_outer.out`，启动前必须均不存在。
- 正式launch恰1次，retry=`NO`。启动后立即核验launcher PID、12个子PID、CWD/cmdline/run-root、固定GPU映射、日志增长和无异常指纹。
- 停止仅限P0协议/安全错误、覆盖风险或至少两个独立候选在有效checkpoint前出现相同确定性异常；必须只停止本run已核实进程树并保留全部工件。绝不按accuracy、loss或其他性能停止。

## 5. 预期工件与后冻结评测

- 每臂必须产生`final_ssdg.pth`、`phase1_clic_terminal_receipt.json`、完整日志及PID/GPU绑定；12/12工件闭合前不得作性能晋级结论。
- 训练完成后运行同一候选的postfreeze导出和目标域盲态评测。每个实验指标都必须包含三种`LEO weak`场景；registered与unknown各自每个物理样本仅用一份预固定received_i，禁多场景择优/融合/重采样。
- 域泛化报告known target的overall、min-class、min-receiver、min-day；未知拒识报告AUROC/AUPR-out/FPR95、unknown FAR/safe rejection、registered false-reject/defer及最差unknown TX/RX/day。defer不计unknown拒识分子。
- 与历史ADV3B02比较不要求同一目标封存包或相同received_i字节；只要求双方训练数据语义配置与known测试数据语义配置分别相同，并由各自不可变工件重开验证。

## 6. 运行回填

- 复用release与启动时间：待回填。
- launcher/PID/GPU/日志：待回填。
- 首波技术健康：待回填。
- checkpoint/terminal计数：待回填。
- 训练完成状态与后冻结工件：待回填。
- GPU/PID/SSH清理：待回填。
- 结果表与最终判定：待工件完整后回填；运行中不读取性能。
