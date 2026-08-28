# D3+ERBT-IDR嵌套新类矩阵实验报告（r5 new1修复）

- 状态：`LOCAL_VERIFIED`
- run ID：`stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r5_new1fix`
- Git提交：`788e559a995fe12356dc1dc03f7fe99d63fa44f0`
- 科学声明：`DIAGNOSTIC_NON_FORMAL`
- 候选：ADV3B02 CORE90+D3 SF-TAPFT+`M29-FFT96-A4/D92-E0-NORF32`
- 矩阵：receiver=`20-1`、data seed=`713101`、method seed=`392002`、K=10、三种`leo_*_weak`场景、`N={1,2,3,5,10,15,20}`，共21格四状态prediction
- 输入：复用r1的`p2_min_v1/VALIDATED_ONCE`嵌套数据与r3三个已完成D3 delta；不重建数据、不重训D3
- 本次修改：D92移除`new5/10/20`类数硬编码，仅保留平衡、有限、每类恰好K条的支持集约束；new1新任务协方差由10条真实support的类内残差自动收缩估计，不复制样本、不改变K、不启用RF32
- 本地验证：新增测试先复现r4同指纹异常；修复后D92/ERBT聚焦回归27/27通过；独立P0/P1审查`APPROVE`
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`，Conda Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：GPU0–7，每GPU最多2格；首个真实checkpoint无query support smoke通过后发布矩阵
- 输入根：r1的`runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r1/input`
- D3根：r3的`runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r3_isolated/d3`
- 输出根：`runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r5_new1fix`
- 预期artifact：每格`support_state_receipt.json`、`predictions.npz`、`prediction_receipt.json`和truth-last`score.json`
- 停止规则：协议/query泄漏、错误row、覆盖、错误代码根、无prediction闭合或至少两格重复确定性异常；低性能不得停止

## 进度

本地修复已提交并推送；N607直接preflight通过，8张RTX3090空闲；r5输出根不存在，r1共84个嵌套输入artifact与r3共3个D3 delta只读核对齐全。
