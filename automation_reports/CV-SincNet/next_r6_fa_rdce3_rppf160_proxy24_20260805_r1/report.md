# NEXT-R6 FA-RDCE3×RPPF160 Proxy24实验报告

## 1.身份与状态

- run ID：`next_r6_fa_rdce3_rppf160_proxy24_20260805_r1`
- 日期：2026-08-05
- 当前状态：`DESIGN_FROZEN / IMPLEMENTING / NOT_LANDED / NOT_LAUNCHED`
- 候选：`NEXT-R6-FA-RDCE3-RPPF160`
- 主agent：`gpt-5.6-sol/high`
- 科学核心、联合runtime与独立审查：不同`gpt-5.6-terra/max`agent
- 后续唯一N607 runner：全部实现、命令与路径冻结后使用`Luna/max`

## 2.假设与因果比较

K5 FA-RDCE3在NEXT-R4中提高域适应后/新类注册后proxy H 1.852个百分点，RPPF160尝试通过全类原型Gram的正则极分解降低类原型相关性。候选不混合qKNN与高斯残差logit，不拟合160×160协方差，也不使用old/new角色分支。

|比较|定义|通过方向|
|---|---|---|
|DA|`R1Q−R0Q`|H_proxy与总正确数增加，retained、held-proxy、floor不降|
|Lite|`R0L−R0F`|同上|
|联合替换|`R1L−R1F`|同上|
|直接实用性|`R1L−R1Q`|同上|

任一完整主比较失败即关闭RPPF160；不调λ、receiver、class、K、量化或矩阵，不重跑。

## 3.冻结方法

RPPF定义：`p_c=unit(mean_k(unit(z_ck)))`，`P=[p_c]`，`G=PᵀP`，`U=P(G+I)^(-1/2)`，`score_c(q)=unit(q)ᵀU_:c`。实现固定FP64对称特征分解，逐列`INT8[160]+FP16 scale`，部署状态`162C`B。K1中FA旁路但RPPF功能化；K5中FA保持NEXT-R4公式与6B动态状态。每个K5 DA×REG状态用自身support cache拟合RPPF，FA状态从REG0到REG1逐bit复用。

历史Full160仅作F臂机制对照，不属于部署候选。详细方法锁见`analysis/next_r6_rppf160_design_20260805.md`。

## 4.冻结矩阵与指标命名

```text
receiver={1-19,14-7}
held-class={14-10,14-7,20-15,20-19,6-15,8-20}
K={1,5}
2×6×2=24 matched conditions
每条件2 REG×6 arms=12 state-arm surfaces
总计288 surfaces
```

receiver由D130清单顺序排除NEXT-R4已评分的`1-1、18-2`后确定，不依据性能选择。四状态统一为`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`。REG0的held-proxy准确率和H_proxy写`N/A`；本矩阵是source-held LOCO，不把held类称为正式Y_new。

K1必须以alias receipt证明`R1Q≡R0Q、R1F≡R0F≡Q、R1L≡R0L`，但L不得alias Q。K5完整执行六臂`R0Q/R0F/R0L/R1Q/R1F/R1L`。

## 5.本地文件与验证状态

|文件|用途|当前状态|
|---|---|---|
|`analysis/next_r6_rppf160_design_20260805.md`|冻结数学、量化、矩阵、资源和裁决|已冻结|
|`analysis/next_r6_joint_da_d92_traceability_20260805.md`|持续目标逐项追溯与三轮回顾|已更新|
|`code/cvsrffi/stage2_next_r6_rppf160.py`|RPPF科学核心|实现中|
|`code/cvsrffi/stage2_next_r6_matrix.py`|24条件/288 surface计划|实现中|
|`code/cvsrffi/stage2_next_r6_runtime.py`|FA×Q/Full/RPPF联合runtime|实现中|

设计级独立复核在纳入per-state refit、`λ=1`和K1 alias后为`P0=0、P1=0、DESIGN_FROZEN`。代码级测试、真实checkpoint smoke与独立审查尚未完成。

## 6.发布前必要项

当前报告不授权N607 handoff。必须补齐：

1.RPPF核心、联合矩阵/runtime和聚焦协议负测；
2.receiver`1-19、14-7`的FA-RDCE3 Phase1资产与量化parity；
3.复用既有VALIDATED_ONCE received-IQ的真实checkpoint无truth smoke；
4.独立代码复核`P0=0、P1=0`；
5.Git提交、文件SHA、本地到远端映射；
6.N607 preflight、GPU、精确CWD/命令、不可覆盖output/log、PID和expected artifacts。

发布停止规则只允许协议/安全违规、错误checkout/hash、输出覆盖、prediction闭合缺失或至少两个不同row在prediction前出现同一确定性异常指纹。不得按accuracy、H、floor或中间性能停止。

## 7.待冻结服务器字段

|字段|状态|
|---|---|
|Conda/Python|待preflight确认；预期`CVS-RFFI`环境|
|CWD|待本地release archive冻结后填写|
|GPU|待preflight后分配，不超过每GPU两个训练任务|
|exact command|未冻结，不得启动|
|log/output/PID|未冻结，不得启动|
|expected artifacts|plan、prediction、manifest、resource、completion、truth-open、score、coverage与alias receipts|

## 8.证据边界

设计和本地测试不是性能证据。只有完整288 surface prediction封存、独立truth-side score和同row资源receipt返回后，主agent才能裁决。即使通过，本run也只产生source-held方向性联合候选，不构成正式Target或多seed推广声明。
