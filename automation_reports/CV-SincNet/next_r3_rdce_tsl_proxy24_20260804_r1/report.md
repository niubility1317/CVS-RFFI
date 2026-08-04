# NEXT-R3 RDCE×TSL-160 Proxy24实验报告

## 1. 基本信息

|字段|值|
|---|---|
|run ID|`next_r3_rdce_tsl_proxy24_20260804_r1`|
|状态|`LOCAL_VERIFIED / N607_ASSETS_READY / PREPARE_PENDING`|
|日期|2026-08-04|
|主agent|Codex主agent（Sol/high）|
|科学实现/审查|Terra/max分工；runner第三次独立审查`P0=0、P1=0、READY`|
|机械实现/发布|Luna/max sole runner；preflight和只读inventory已完成，尚未sync/launch|
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
|`f35c3cdf`|隔离`prepare→predict→score`runner；真实checkpoint bridge与共同query|

分支中夹有他人并行D92/D138提交；本报告不把它们计入NEXT-R3实现证据，也不得覆盖其工作树改动。

### 5.2 已完成验证

- 独立代码审查初轮：`P0=0、P1=3`。
- 三项P1修复后二次审查：physical-LOO、共同旧query字节绑定、runtime→artifact→score均`CLOSED`；最终`P0=0、P1=0`。
- 在commit`19b0650f`的独立干净worktree、`ssr-gpu`环境串行执行NEXT-R3五组测试和D129 heads回归：`28 passed`。
- 临时验证worktree已删除。

### 5.3 runner独立复核与最小修复

commit`f143e5d2`的runner独立复核结论为`P0=2、P1=2、NOT_READY`。两项P0分别是：实际24行仍直接消费外部`source-held pre_relu`，真实checkpoint只承担重复smoke，尚未证明预测特征由received-IQ经checkpoint生成；predict阶段读取`tx_labels`并据此构造query集合，违反query truth/role禁入。两项P1是K1/K5共同旧query未在runner入口闭合，以及新增测试只覆盖缺失输入和run-root不可覆盖。当前只修这些直接阻止真实性能实验的问题；矩阵、候选和性能阈值不变，不追加工程gate。修复并再次独立得到`P0=0、P1=0`前，不进行remote inventory、sync或launch。

commit`7f0fb9a4`已关闭真实特征输入问题：正式support/query/Phase1特征均改由received-IQ和绑定checkpoint的bridge生成，聚焦测试`5 passed`。但第二次独立复核仍为`P0=2、P1=1、NOT_READY`：两份REG0/REG1 query ID清单的差集会暴露新类角色；predict仍能读取全部588条`class_ids/tx_labels`。因此当前改为复用既有`prepare→predict→score`隔离：prepare封存Phase1 prior、合法K-shot support和单一共同query，predict不接触全量标签或query角色，旧/新/H划分只在完整prediction之后由score打开truth完成。

commit`f35c3cdf`完成隔离后，第三次独立复核为`P0=0、P1=0、READY`。`predict`只接收received-IQ、checkpoint和无truth predictor package；REG0/REG1及K1/K5逐字复用同一共同query。主agent在`ssr-gpu`环境串行执行runner、NEXT-R3核心和D129 head回归，共`40 passed`，`py_compile`通过。

## 6. 真实输入现状

N607只读inventory已确认既有真实资产足以执行一次`prepare`并生成本run专用predictor package和truth；不复用旧D104 truth，不重建数据。

|资产|绝对路径|SHA256/合同|
|---|---|---|
|received-IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`；`<f4[588,2,256]`|
|IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`；validator=`2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc`|
|Phase1 cells/strict tap|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；588行|
|tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|RDCE wire|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`|`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|

source-held proxy的authority绑定使用既有不可变证据：`capsule_id=received-IQ archive SHA`，`split_id=scorer-only split manifest SHA 6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`，`validator_receipt_sha256=2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc`。这些值只标识本次`SOURCE_HELD_PROXY`输入，不构成正式Target新类注册声明。

## 7. N607发布预注册

2026-08-04 20:12 CST已完成一次只读direct preflight：目标`N607`解析为`172.31.111.215`，普通用户`szu2070436088`身份通过；远端`dell-DSS8440`及项目根`/home/szu2070436088/2510044040/CV-SincNet`可见；8张RTX 3090均可见且当时利用率为0%、显存约1MiB。命令退出后本地`ssh.exe=0`，到`172.31.111.215:22`的`ESTABLISHED=0`。该证据只表示连接与资源可见，不表示runner已可发布。

|字段|预注册值|
|---|---|
|local commit|`f35c3cdf8737068f5aca2e9b2ddcb164a7579bf2`|
|source sync mapping|本地`release/next_r3_runtime_f35c3cdf.tar.gz`→remote`input/next_r3_runtime_f35c3cdf.tar.gz`；SHA=`151c0f1d76ad8b0d373e318fdef38149d412980ae1bbca5500662bc4bfc01abf`；6,475,647B|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r1`|
|working directory|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r1/source`|
|environment|`ssr-gpu`|
|exact command|见下方三段；`prepare`、单行smoke、完整24行严格顺序执行|
|GPU allocation|物理GPU0；`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`|
|stdout/stderr log|`.../logs/prepare.log`、`.../logs/smoke.log`、`.../logs/run.out`、`.../logs/run.err`|
|PID receipt|`.../logs/main.pid`|
|prediction|`.../output/prediction.json`|
|manifest/resource/smoke|`.../output/{manifest,resource,smoke}.json`|
|truth-side score|`.../output/score.json`，仅prediction 24/24完整后生成|

`prepare`命令固定使用上述received-IQ、receipt、strict tap和checkpoint，输出`.../prepare/{predictor_package,truth,prepare_receipt}.json`；`capsule-id=e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`、`split-id=6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`、`validator-receipt-sha256=2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc`。生成后按字节计算package SHA并传入`predict`；单行smoke使用新目录`.../smoke`，完整矩阵使用新目录`.../output`。两次predict均固定同一received-IQ/checkpoint/tap/RDCE wire及上述SHA；完整prediction闭合后，`score`才读取本次`prepare/truth.json`并写入新文件`output/score.json`。

### 7.1 精确执行命令

以下变量在同一个有界远端命令中固定展开：

```bash
ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r1
PY=/home/szu2070436088/.conda/envs/ssr-gpu/bin/python
IQ=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz
IQ_RECEIPT=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json
TAP=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz
TAP_RECEIPT=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.receipt.json
CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth
RDCE=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire
```

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" code/scripts/run_next_r3_proxy24.py prepare --output-dir "$ROOT/prepare" --received-iq "$IQ" --received-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --received-iq-receipt "$IQ_RECEIPT" --received-iq-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --phase1-cells "$TAP" --phase1-cells-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint "$CKPT" --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --capsule-id e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --split-id 6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134 --validator-receipt-sha256 2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc --device cuda:0 >"$ROOT/logs/prepare.log" 2>&1
PACKAGE_SHA=$(sha256sum "$ROOT/prepare/predictor_package.json" | awk '{print $1}')
CUDA_VISIBLE_DEVICES=0 "$PY" code/scripts/run_next_r3_proxy24.py predict --run-id next_r3_rdce_tsl_proxy24_20260804_r1-smoke --run-root "$ROOT/smoke" --received-iq "$IQ" --received-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --received-iq-receipt "$IQ_RECEIPT" --received-iq-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --package "$ROOT/prepare/predictor_package.json" --package-sha256 "$PACKAGE_SHA" --checkpoint "$CKPT" --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --d106-tap-archive "$TAP" --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --d106-tap-receipt "$TAP_RECEIPT" --d106-tap-receipt-sha256 24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665 --d106-rdce-wire "$RDCE" --d106-rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --device cuda:0 --smoke-one-row-no-truth >"$ROOT/logs/smoke.log" 2>&1
nohup env CUDA_VISIBLE_DEVICES=0 "$PY" code/scripts/run_next_r3_proxy24.py predict --run-id next_r3_rdce_tsl_proxy24_20260804_r1 --run-root "$ROOT/output" --received-iq "$IQ" --received-iq-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --received-iq-receipt "$IQ_RECEIPT" --received-iq-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --package "$ROOT/prepare/predictor_package.json" --package-sha256 "$PACKAGE_SHA" --checkpoint "$CKPT" --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --d106-tap-archive "$TAP" --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --d106-tap-receipt "$TAP_RECEIPT" --d106-tap-receipt-sha256 24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665 --d106-rdce-wire "$RDCE" --d106-rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --device cuda:0 >"$ROOT/logs/run.out" 2>"$ROOT/logs/run.err" & echo $! >"$ROOT/logs/main.pid"
```

完整prediction闭合后执行：

```bash
TRUTH_SHA=$(sha256sum "$ROOT/prepare/truth.json" | awk '{print $1}')
"$PY" code/scripts/run_next_r3_proxy24.py score --run-root "$ROOT/output" --truth "$ROOT/prepare/truth.json" --truth-sha256 "$TRUTH_SHA" --output "$ROOT/output/score.json"
```

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
