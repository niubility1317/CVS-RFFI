# Phase1 CLIC真实训练入口三批技术烟测报告

## 1. 状态与目标

- 实验ID：`phase1_clic_smoke3_20260812_v1`
- 当前状态：`PREREGISTERED / NOT_LANDED`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 目标：在N607真实ManySig、真实GeoSat-C final checkpoint和真实训练器入口上，分别对C、G执行恰好3个optimizer batch；三个batch依次使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，验证数据元信息、单份received-IQ弱星地信道、CLIC前向、共同L_base、VJP、AMP、资源回执和图释放的完整闭环。
- 本烟测只判断技术健康，不读取或比较准确率、loss优劣等性能，不产生checkpoint、terminal、prediction或score，也不用于选模。

## 2. 根因与修复

- v4正式矩阵12/12在首batch同一错误`P1-CLIC source-L batch metadata is absent`退出；原因是训练器把真实`move_batch`返回的`extra=(domain, metadata)`错误当成直接metadata映射。
- 修复后通过既有`_meta_from_extra`读取第二项metadata，严格提取128行`base_index/sig_i/local_label`；仅接受整数，拒绝bool、浮点、字符串、非有限值、负base/sig和越界标签。
- 新增参数`--phase1_clic_technical_smoke_batches 3`。它复用正式训练完整batch生命周期，在三场景全部审计、AMP和图释放闭合后写技术receipt并退出；默认`0`保持正式40epoch路径不变。
- 发布核心commit：`f43f313e`（`fix: bind CLIC trainer batch metadata`）。

## 3. 冻结烟测矩阵与配置

| 行 | 折 | 臂 | operator | GPU | batch | 场景顺序 |
|---|---:|---|---|---:|---:|---|
| `F1C_CLIC12` | F1 | C | `raw_phase_control` | 0 | 3 | clear→low_elev→rain |
| `F1G_CLIC12` | F1 | G | `complex_local_invariant_curvature` | 1 | 3 | clear→low_elev→rain |

- 其余配置完全继承冻结launcher：seed=`7281164`、batch=`128`、AdamW、`clean CE+0.10×KL(clean-stopgrad→single-LEO)`、4个source-L训练TX、1个known-validation TX、1个proxy-unknown TX；held/proxy/query/target训练访问均为0。
- G只从同一份received_i提取lag=`{1,2,4,8}`的多尺度三点复曲率token；C/G除operator外配置相同。
- 唯一新增命令参数为`--phase1_clic_technical_smoke_batches 3`；run/log/output均使用本烟测不可覆盖ID。

## 4. 本地证据

- TDD RED：15组非法物理绑定中14组按预期失败；旧实现会静默截断bool/浮点/字符串或放行负值与越界标签。
- GREEN：物理绑定定向19项通过。
- `ssr-gpu`：`py_compile train_ssdg.py/test_phase1_clic.py`通过；`test_phase1_tx_partition.py + test_phase1_clic.py`共195项通过，仅既有AMP弃用warning。
- `git diff --check`通过，仅工作树LF/CRLF提示。
- 独立复审：`SPEC=PASS / QUALITY=PASS / P0=0 / P1=0`。
- 核心文件SHA256：`train_ssdg.py=33371D4504F08037BF1345A245826107615391A706DD05F04E6DD1E126B10A08`；`test_phase1_clic.py=8DDFCFC7F2C0C025B5F92C89B0A24AA80757888C8B908772B278D8693DF8B697`。

## 5. N607发布与停止合同

- 发布源必须是Git commit `f43f313e`的干净archive，不得携带未提交Task7文件。
- 预定release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_smoke3_20260812_v1_f43f313e`。
- 预定run/log：`runs/phase1_clic_smoke3_20260812_v1`、`logs/phase1_clic_smoke3_20260812_v1`；落地前必须均不存在。
- 先执行普通账户直连preflight、GPU/进程检查、archive与远端文件hash、`py_compile`、trainer help、launcher dry-run；随后仅启动上述C/G两行。
- 成功条件：两份`phase1_clic_technical_smoke_receipt.json`均`completed=true`、`batches=3`、三场景各1、VJP/AMP/resource/graph release闭合，且proxy/query/target/selection均0；无`final_ssdg.pth`、无terminal receipt。
- 失败条件：任一配置/协议/执行异常立即封存烟测为技术失败，不启动正式v5；不得在同一run ID重试。任何性能值都不得作为停止或启动依据。

## 6. 运行回填

- archive/SCP/release：待回填。
- C/G PID、GPU、日志：待回填。
- 技术receipt：待回填。
- checkpoint/terminal/prediction/score计数：待回填。
- GPU/PID/SSH清理：待回填。
- 最终结论：待回填。
