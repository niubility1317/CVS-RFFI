# NEXT-R3 RDCE×TSL-160 Proxy24实验报告

## 1. 基本信息

|字段|值|
|---|---|
|run ID|`next_r3_rdce_tsl_proxy24_20260804_r1`|
|状态|`LOCAL_VERIFIED / REAL_INPUT_INVENTORY_PENDING`|
|日期|2026-08-04|
|主agent|Codex主agent（Sol/high）|
|科学实现/审查|Terra/max分工；二次独立审查`P0=0、P1=0`|
|机械实现/发布|Luna/max；N607 sole runner尚未交接|
|协议|`p2_min_v1`|
|证据语义|`SOURCE_HELD_PROXY`；`formal_new_registration_claim=false`|

## 2. 目标、假设与比较对象

本实验只检验一个联合候选：`R3-RDCE160×TSL-160`。RDCE复用D106的rank-3封存资产；TSL-160以Phase1完整physical-LOO封存的source-anchor先验约束K5对角头相对球形参考头的函数位移。假设是：RDCE能保留既有小幅source-held域适应信号，TSL能避免D130无约束Lite160的held/floor下降，并在同row组合后保持正交或互补收益。

比较对象固定为`R0/R1×Q/F/L`：`Q`是冻结qKNN，`F`是同160维role-structured D92-Full160参照，`L`是TSL-160。主因果量为`R1Q-R0Q`、`R0L-R0Q`和`[R1L-R1Q]-[R0L-R0Q]`。`R0L-R0F`与`R1L-R1F`只解释为TSL相对F160的全管线替换差，不称为纯head效应，也不替代历史formal288 D92。

## 3. 冻结矩阵与四态命名

|维度|冻结值|
|---|---|
|held receiver|`1-1、18-2`|
|held class|6个显式唯一类，由真实archive注册表绑定|
|K|`K1、K5`|
|原子行|`2×6×2=24`|
|每REG底层bundle|`R0Q/R0F/R0L/R1Q/R1F/R1L`六臂|
|四态|24行×4=`96 state`|
|唯一state-arm|96×3=`288`|

|状态|含义|指标边界|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|旧类、old-floor、共同旧query总正确数；N/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|同上；N/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|旧类、held-proxy新类、H、全注册floor、总正确数|
|`DA1_REG1`|冻结同一RDCE状态后的域适应表示＋注册|同上|

所有artifact和结果表必须标注`SOURCE_HELD_PROXY`。held class不是正式Target新类；本实验不能产生Target/125或正式新类注册声明。

## 4. 方法与资源冻结

### 4.1 RDCE

- 不可变数值资产：504B；每row动态attenuation：6B。
- 每query额外投影：960MAC。
- K1固定`a=(0.3,0.3,0.3)`；K5只用REG0旧类support的类等权类内scatter闭式拟合。
- `DA1_REG1`必须复用`DA1_REG0`同一state SHA；只编码新增support，禁止重拟合DA。

### 4.2 TSL-160

- Phase1 prior：170B；每个outer fold同时排除held receiver和held class。
- eligible physical IDs必须逐一single-holdout，validation恰好覆盖一次，support是严格补集。
- R0消费D106 canonical normalized-ReLU缓存；R1直接消费RDCE signed unit缓存，禁止再次ReLU或归一化。
- `prior_semantics=pre_adaptation_source_anchor_same_ambient_axes`、`prior_transported_by_rdce=false`、`r1_covariance_claim=false`。
- 仿射部署状态：`164C B`；K5 fit MAC：`4N×160+8×160+2C×160`；query：`160C MAC`。
- K1的F/L逐logit精确别名Q，不宣称head收益。

## 5. 本地版本与验证

### 5.1 相关提交

|commit|内容|
|---|---|
|`6f402e4a`|TSL-160核心与初始测试|
|`9fb9cb93`|RDCE-R1 signed cache与source-anchor绑定|
|`2648cf1a、de8dc2ba、9eb3386f`|24行matrix/scorer及96态/288臂语义修复|
|`44dafb4d、c74c76b6`|RDCE四态runtime与共同旧query逐字节绑定|
|`22f4e54e`|方法锁与设计追踪|
|`51878e24`|完整physical-LOO覆盖|
|`19b0650f`|truth-free runtime→artifact闭环|

分支中夹有他人并行D92/D138提交；本报告不把它们计入NEXT-R3实现证据，也不得覆盖其工作树改动。

### 5.2 已完成验证

- 独立代码审查初轮：`P0=0、P1=3`。
- 三项P1修复后二次审查：physical-LOO、共同旧query字节绑定、runtime→artifact→score均`CLOSED`；最终`P0=0、P1=0`。
- 在commit`19b0650f`的独立干净worktree、`ssr-gpu`环境串行执行NEXT-R3五组测试和D129 heads回归：`28 passed`。
- 临时验证worktree已删除。

## 6. 真实输入现状

本地Git快照只有receipt/fixture/manifest，没有真实`d106_ls_received_iq.npz`、strict-tap/features NPZ、D106 RDCE asset wire或checkpoint PTH。因而当前不能完成real-checkpoint smoke，也不能本地构建每fold完整TSL prior。缺失的最小真实字段为：

1. `received_iq`及physical ID、receiver、class绑定；
2. 每个合法Phase1物理样本的canonical normalized-ReLU 160维行；
3. D106 formal RDCE asset wire及SHA；
4. 真实checkpoint及SHA；
5. 与上述字段一致的split/tap/physical-root/seal receipts。

禁止使用合成数组、receipt占位或不完整fold把real smoke标为通过。下一步只在N607做短连接只读inventory，确认这些既有资产的精确路径；不重建数据、不启动实验。

## 7. N607发布预注册

|字段|预注册值|
|---|---|
|local commit|`PENDING_FINAL_RUNNER_COMMIT`|
|source sync mapping|`PENDING_RUNNER_AND_INVENTORY`|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r1`|
|working directory|`PENDING_RUNNER_AND_INVENTORY`|
|environment|`ssr-gpu`|
|exact command|`PENDING_RUNNER_AND_INVENTORY`|
|GPU allocation|`PENDING_N607_PREFLIGHT`|
|stdout/stderr log|`.../logs/run.out`、`.../logs/run.err`|
|PID receipt|`.../logs/main.pid`|
|prediction|`.../output/prediction.json`|
|manifest/resource/smoke|`.../output/{manifest,resource,smoke}.json`|
|truth-side score|`.../output/score.json`，仅prediction 24/24完整后生成|

任何占位字段在sole runner交接前必须替换为真实值；占位报告不能授权launch。

## 8. 健康停止与成功条件

健康停止只允许两类触发：P0协议/安全违规；或至少两个不同row在产出prediction前出现相同确定性exception fingerprint。停止前必须绑定本run的PID/CWD/cmdline/run-root，保存partial artifacts；不得按H、准确率、floor或任何性能值早停。

技术成功条件：真实checkpoint one-row no-truth smoke通过；24/24 row、96/96 state、288/288 arm prediction完整；query fit/update/selection全为0；truth只在完整prediction之后打开；输出路径未预存在且未覆盖。只有满足这些条件才进入性能分析。

## 9. 完成后结果表模板

|状态/比较|K|receiver|old/retained BA|held-proxy BA|H|old floor|all floor|总正确数|资源/时延|结论|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|`DA0_REG0`|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|`DA1_REG0`|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|`DA0_REG1`|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
|`DA1_REG1`|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
|`R1Q-R0Q`|K5|pooled|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|RDCE 510B+960MAC/query|PENDING|
|Q/L DID|K5|pooled|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
|`R0L-R0F`|K5|pooled|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|F参照，不是纯head效应|
|`R1L-R1F`|K5|pooled|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|F参照，不是纯head效应|

## 10. 结束后更新

完成后必须在本节填写：最终状态；remote hash/PID/GPU/SSH断开证据；24行详细same-row表；K/receiver/class分层；拟合墙钟与峰值工作集；异常；是否因完整负证据关闭，或是否只允许进入下一阶段。不得用边际最大值替代联合行，也不得把partial/technical smoke写成性能结果。
