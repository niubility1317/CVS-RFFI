# D55中心化LOO难度截距补偿报告

## 1.状态与目标

- 状态：`IMPLEMENTED_AND_TESTED_PRE_RUN`；operator Codex。
- receiver20-1、seed713101、K10/new5、3场景×5fold；本地、无N607、无125。
- 目标：在D46上仅补偿support-LOO困难类，联合改善rain old与low-elev new floor。

## 2.公式、协议与验证

`d_c=sum_g w_gc CE_gc`，`Delta b_c=d_c-mean(d)`，`W_D55=W_D46`，`b_D55=b_D46+Delta b`。无系数/温度/阈值/clip/扫描/类ID/角色/scene/receiver/query；K1/K2精确D46 fallback。复用`VALIDATED_ONCE p2_min_v1`，support-only，clean/source/query truth/quota/count/global assignment禁止。

D55定向7项、D46＋D55联合23/23、`py_compile`通过；额外适配仅136 MAC-equivalent、0比较。成功须保持D46 after/new/min-new并改善H/forget/joint/floor，且无场景交换伤害；否则停止，不跑第二seed/formal/125。

完成后必须报告7候选、3场景、逐类、15fold、matched版本、20epoch、混淆、补偿分布、量化、资源、artifact SHA。
