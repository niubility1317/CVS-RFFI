# PAMF 物理配对训练后的冻结身份结构诊断

## 预登记

Run `20261030-phase3-star-physical-geometry-s2026092701-r01`；执行代码 `8aeac9583dc6becf4bcfd7ccebea06a1ee6176a8`。固定原 F0 fused160、P_L fused160/open64、主 P_LOE fused160/open64，共五种表征；不训练、不换主模型、不挑权重或阈值。方法见 `docs/PAMF_PHASE3_PHYSICAL_GEOMETRY_PLAN_20261030.md`；唯一行矩阵见 experiment.json。

仅复用 source novel 3150 条固定 G10P1：三个未知设备各 1050 条，source 五接收机×三日期×每设备每域 70 条。接收机为 1-19、18-2、19-2、2-19、3-19，日期为 2021-03-08/15/23。未知 TX 19-15/19-16/7-1 只由独立评分端读取。源接收机已见、源未知设备有历史开发暴露，不宣称新 RX/TX 盲测。新增 L/V/OE/Us/target U/Q/正式 RX24-18 IQ 均为 0，P0/clean 新值读取为 0，G 加载及前向为 0；后 Phase1 输入始终为星地 P1。

两个父 checkpoint 均为原 physical-open 源训练 2220 新增／累计 13320 步、代码 31fa172946aeaecd2d3f1e784aee7366b7b5bee1，源数据契约及原 F0/DURA 祖先和缓存一致。F0 E50/11100 的同池 unit fused160 直接复用旧诊断导出代码 55977eec2cf0c2bd09d7c6856179f919e9f0923d；检查身份、source manifest、逐值 IDs/RX/day 对齐，不加载 F0、不重复前向。

两个真实 checkpoint 各 2 条合成无 query smoke 在真实 source role 打开前执行；实际 frozen state 逐次检查。同次前向导出 fused160/open64，不另训头、不更新参数、阈值、原型或 buffers。新增 F 前向 6300 实物行＋4 合成行；F0 特征复用 3150 行；独立物理记录仍为 3150。

五种完整特征导出后，独立 CPU 先固定每种 all-other/cross-RX/cross-day/cross-RX-and-day 共 20 套 1NN 索引。排除自身，FP64 cosine、blocked128、opaque ID 稳定处理 ties，TX/K/配额不能进入选择。全部选择固定后才唯一读取 source truth；只报告同池宏命中率及逐 RX/day/TX、类内外 cosine gap、有效秩／参与秩、TX/RX/day 组间方差比。真零保留、缺候选 N/A；方差比非正交或因果解释。

这些指标不等于无标签分类、未来匿名注册 A_N 或 Phase3 成功。主 P_LOE 身份固定，eligiblefalse/DIAGNOSTIC_ONLY，AN/HOW/UDA/REG N/A。v3 的 CK90/RU90、A_K 距同输入原 F0 闭集不超过 2 个百分点及未来注册目标保持 ACTIVE。

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-physical-geometry-s2026092701-r01-release`；owner `codex-phase3-physical-geometry-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_physical_geometry_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-physical-geometry-s2026092701-r01-release/runtime.json`。
- 独占 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-physical-geometry-s2026092701-r01`；只新增一个 GPU export worker，actualfree≥4096MB，每卡最多两个本任务新增进程；独立 CPU scorer 的 CUDA 为空、BLAS1。其他用户停止、修改、重启为 0。
- 聚焦 IO 5 PASS/5.61s、score 4 PASS＋26 subtests/1.90s；5 AST 及实际 runtime 接口通过，唯一直接 P0/P1 审查 PASS。一次 N607 preflight 已通过，只读句柄、父身份、文件存在及显存，IQ/truth 值读取 0；后续单 archive SHA 比较／远端 compile、实际首步 smoke、独立 PID/CWD/argv/GPU/log 增长核验。
- 产物为 launch/pipeline/config/log；features F0/P_L/P_LOE NPZ、smoke/complete；score results/complete；observations。数据、特征、权重和原日志不提交 Git。
- 非法角色／祖先、非有限、输出碰撞或不可预测技术失败时保留，不自动重试；UNKNOWN 先核原 PID/root，不操作他人任务。此刻仅完成预登记，尚未宣称启动或得到结果。
