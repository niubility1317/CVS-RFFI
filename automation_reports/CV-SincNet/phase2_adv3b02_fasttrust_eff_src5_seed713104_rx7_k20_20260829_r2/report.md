# Phase2七接收机域目标确认报告r2

## 预登记

- run_id：`phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r2`
- 状态：`LOCAL_VERIFIED`
- 候选/checkpoint：冻结`S713104_ADV3B02_FASTTRUST_EFF`的`final_ssdg.pth`
- 方法：`freq_f3_proj`，1步support-only late-block adaptation，`learning_rate=0.0005`，可训练比例`[0.03,0.15]`
- 数据：`p2_min_v1`、`VALIDATED_ONCE`、capsule`536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`、split`260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25`
- 矩阵：`K=20`，7接收机×3场景=21行，每行1352个query，26类；仅报告`DA1_REG1`，其余三状态`NOT_RUN`。
- prediction-first：全部21行原子prediction完整验证后才允许独立scorer连接truth。
- Git提交：`012968d2cff58ab3b2ffaf19157030212341b9f1`
- 本地验证：49项聚焦测试、编译、CLI和`git diff --check`通过。
- release：`release_v5.zip`，SHA256`3dfa6eef113a1e04afbdddd7f3689f69dd6852d96ad295deeea68ce73993546e`
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r2`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r2`
- GPU：接收机`1-1,14-7,2-1,20-1,7-14,7-7,8-8`分别绑定GPU0–6，每个进程顺序运行3个场景，GPU7保留。
- 停止规则：仅协议/路径/输出碰撞、确定性执行异常、无合法prediction闭合或scorer连接错误；不得因性能停止。
- r1关系：r1因连续预prediction系统技术失败停止，0/21 prediction、0次truth连接；r2不复用r1输出，只复用同一已验证canonical数据和冻结checkpoint。

## 正式结果

待prediction-first与独立评分闭合后填写。
