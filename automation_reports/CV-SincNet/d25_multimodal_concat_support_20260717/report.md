# D25多表征拼接support-only原子筛选

## 启动前记录

- experiment ID：`d25_multimodal_concat_support_20260717/support_screen_v1`
- 日期：2026-07-17；operator：Codex
- 状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS`，尚未同步或启动N607。
- 目标：在不打开query的前提下，使用同一密封LEO_weak enrollment-only support，对D25的288维分块拼接、不确定度ground-z融合和逐块半径评分执行15fold原子筛选。
- 假设：保留`z_id160+FFT96+RF32`完整288维，同时把块平方能量从D1的辅助分支94.12%支配修正为按维数比例`5/9、1/3、1/9`，可以保留多表征平均增益并改善旧类与新类floor稳定性。
- 对比：`Z0_SUPPORT_ONLY`、`B3_SINGLE_IQ_DIAG_FFTRF`、`D25_C0_DIM_CONCAT`、`D25_C1_UF_GROUNDZ`、`D25_C2_BLOCK_RADIUS`。
- 筛选规模：5候选×3个互斥LEO_weak场景×5个held-rank fold=75条support-fold结果。

## 协议边界

- 复用D22 v4的before/after `enrollment_only`密封support package、seal、formal policy、authorization、signed envelope、class binding和历史int8组件。
- 历史int8组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，仅用于用户授权的`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`；本实验不获得formal authority或正式性能声明资格。
- 每个物理support只有一个已叠加一次且仅一次LEO_weak信道的IQ观测。
- `z_id160`、`FFT96`、`RF32`是同一received-IQ的一组三个数学feature blocks，最终只形成一条288维行；`support_view_count=1`、`support_row_multiplicity=1`、`derived_support_rows=0`。
- runner CLI和运行时不接收query root、query seal、truth、role、真实query批次类别数、类别quota、global assignment或scorer输入。
- 不访问clean/source样本、sample-level source/full-precision feature或clean-derived信号。
- held support仅用于预登记leave-two-out确认，不是正式query；结果只能用于support-only方法锁。

## 本地版本与实现

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录`E:\type10-7`不是Git仓库，本报告同步维护Git镜像。
- D25核心提交：`f349850d`。
- D25核心SHA256：`c8789679888bee15e9e3167dcdd576458494fd471f5f83b747836720657f75c7`。
- 已有核心：`code/cvsrffi/stage2_multimodal_concat_fusion.py`。
- 待新增：`code/scripts/run_d25_support_only_concat.py`、`tests/test_run_d25_support_only_concat.py`、launcher。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 当前验证：D25+D24+D23共37项PASS；runner focused测试待完成。

## 数据与筛选定义

- receiver：沿用D22开发receiver `20-1`。
- seed：沿用开发seed `713101`。
- K=10；每fold每类fit K=8、held K=2。
- 旧类6个；真实seen-new5个。
- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，三者physical sample/token/received-IQ hash必须两两不交。
- `HELD_RANKS=((0,1),(2,3),(4,5),(6,7),(8,9))`。

## 晋升与停止条件

D25候选相对Z0必须同时满足：

- 15/15fold逐旧类after-old非劣；
- 15/15fold逐新类after-new非劣；
- before/after/new floor、H均非劣；
- forgetting不增加；
- Stage2-C后旧类score列、旧prefix、旧prototype/radius/count逐字节冻结；
- 最差after-old floor或最差joint floor严格改善。

`B3_SINGLE_IQ_DIAG_FFTRF`只作为历史机制的同run诊断对照，不可晋级。若全部D25候选失败，回退Z0并记录负证据，不打开query、不生成正式prediction artifact。

## 资源审计

对每个候选×场景分列：

- trainable parameters、epoch、optimizer steps；
- persistent state、fit scratch、Phase1 int8 logical/serialized bytes；
- backbone forward、FFT96 `O(T log T)`、RF32 `O(T)`及quantile/sort单列；
- concat/head MAC与identity-only单qKNN对照；
- enrollment适配时延、batch=1推理平均/P95、峰值RAM/VRAM；
- no dense query graph、query rows used for fit=0。

## N607计划

- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python：启动前重新确认；历史为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU/PID：待preflight和live inventory后确定。
- 远端runner：`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/run_d25_support_only_concat.py`。
- 远端log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d25_multimodal_concat_20260717/support_screen_v1.log`。
- 远端output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d25_multimodal_concat_20260717/output/support_screen_v1`。
- 精确命令、同步映射、PID和GPU在本地验证、N607 preflight及占用检查后补入。

## 预期产物

- `RECEIPT.json`
- `training_log.jsonl`
- `selection.json`
- `support_audit.json`
- `resource_audit.json`
- `geometry_audit.json`

不得输出原始IQ、样本级特征、FP32 prototype向量、query prediction、truth sidecar或score table。
