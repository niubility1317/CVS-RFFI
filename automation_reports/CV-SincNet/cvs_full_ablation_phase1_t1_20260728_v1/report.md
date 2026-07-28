# CVS-RFFI Phase1第一层全量消融

## 身份与状态

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260728_v1`|
|日期|2026-07-28|
|operator|Codex主代理；N607发布将交给唯一实验runner子代理|
|状态|`BLOCKED_PRELAUNCH_MISSING_N607_SSR_GPU_NOT_LAUNCHED`|
|设计|`CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`|
|协议|Phase1 source-only；正式划分`0.07/0.63/0.30`|
|当前Git分支|`codex/full-ablation-20260728`|
|代码就绪提交|`67b41f6d5eecfa90aaf134c891bd43f2a4793997`|
|独立终审|`P0=0、P1=0`；`APPROVE_LOCAL_VERIFIED`|
|审阅者|`/root/phase1_t1_independent_review`|
|性能结论|无；尚未连接N607、同步或启动|

## 目标与假设

目标是完成`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`六个第一层arm的5个paired seed重训，共30次完整训练。主假设是完整Phase1在source-only UDU、hard-TX、source-side LEO压力和表征泄漏/稳定性上优于组消融，同时不把target receiver指标用于checkpoint选择。

对照关系：

|arm|唯一因果组|对照|
|---|---|---|
|`P1-FULL`|完整方法|参考|
|`P1-SUP`|只保留labeled source与TX/CosFace目标|`P1-FULL`|
|`P1-A0`|参数量精确匹配的单embedding，去显式domain表征/解耦|`P1-FULL`|
|`P1-B0`|只关闭伪标签CE与熵项|`P1-FULL`|
|`P1-C0`|关闭prototype、几何、`L_zid`、`L_coretail`和soft-mix组|`P1-FULL`|
|`P1-D0`|关闭MixStyle、source episode和LEO压力CE组|`P1-FULL`|

## 冻结矩阵

- Phase1 seeds：`7281101–7281105`。
- 每个seed在六个arm间paired，共30个row。
- worker为16个持久槽：GPU0–GPU7，每卡slot0/slot1；第一波最多16个进程，尾波14个。
- 服务器已有训练进程计入每卡2进程上限；调度器在每次启动前读取实际GPU compute PID并扣除本run已拥有PID。
- 任何输出、日志、PID或status路径碰撞均失败关闭。

## 本地实现与验证

|文件|作用|
|---|---|
|`code/cvsrffi/full_ablation_spec.py`|30/75/900矩阵、seed、16槽和artifact schema|
|`code/cvsrffi/phase1_ablation_factory.py`|六arm唯一配置、差分和config hash|
|`code/model_dual_cvsincnet.py`|`P1-A0`参数量匹配单embedding|
|`code/SSDG/train_ssdg.py`|正式`ablation_id`入口、source-validation选模、split与完成收据|
|`code/scripts/build_full_ablation_plan.py`|非启动型不可覆盖计划构建|
|`code/scripts/run_full_ablation_phase1_t1.py`|16槽runner、占用上限和重复异常指纹停派|
|`code/scripts/seal_full_ablation_phase1_plan.py`|独立审查、真实checkout、发布文件SHA与计划封存|
|`configs/full_ablation_20260728/seed_registry.json`|fresh seed/draw注册表|

验证证据：

|检查|结果|
|---|---|
|聚焦规范、arm factory、A0、runner、sealer与收据测试|53项通过|
|A0参数量|与完整双表征模型精确相等；8域测试fixture均为1,061,334，真实14域均为1,062,306；正式值以row resource summary为准|
|训练CLI dry-run|六个arm均解析成功；200轮和`0.07/0.63/0.30`被工厂强制覆盖|
|矩阵dry-run|30 rows、16 slots|
|静态编译|8个正式发布文件通过|
|真实checkpoint no-query smoke|历史封存checkpoint安全加载；0 missing、0 unexpected；输出`[2,6]`、`z_id=[2,160]`且有限|
|已知非本改动失败|`tests/test_adv3b02_supervised_da_runner.py`有2项历史runtime contract字段漂移；未纳入本次通过计数，未修改|

正式checkpoint只由source validation选择并保存为`best_source_validation_ssdg.pth`；target receiver/day/LEO只在选择冻结后评估。PyTorch 2.6把checkpoint加载默认改为安全`weights_only`后，历史artifact中的`SatViewStage`会被拒绝。本次公共加载器只对白名单`SatViewStage`开放安全加载，没有回退到不受限反序列化。

真实checkpoint无query smoke使用：

- 路径：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_strict_dual125_20260714_183556\artifacts\best_joint_safe_ssdg.pth`
- SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- 环境：`ssr-gpu`；CPU前向；不读取dataset、support或query。

精确命令：

```powershell
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate ssr-gpu
$env:PYTHONPATH='E:\type10-7\github_publish\CVS-RFFI-repo\code'
@'
import hashlib, json
from types import SimpleNamespace
import torch
from post_stage_common import build_baseline_model, load_checkpoint, merge_checkpoint_args
p=r"E:\type10-7\automation_reports\CV-SincNet\qknnv42_strict_dual125_20260714_183556\artifacts\best_joint_safe_ssdg.pth"
d=torch.device("cpu")
c=load_checkpoint(p,d)
n=int(c["model"]["dom_head.net.3.weight"].shape[0])
a=merge_checkpoint_args(c,SimpleNamespace(),input_len=int(c["args"]["wisig_out_len"]),num_domains=n)
m=build_baseline_model(a,d)
missing,unexpected=m.load_state_dict(c["model"],strict=False)
m.eval()
with torch.no_grad():
    o=m(torch.randn(2,2,int(a.input_len)),return_aux=True)
r={"checkpoint":p,"checkpoint_sha256":hashlib.sha256(open(p,"rb").read()).hexdigest(),"python_environment":"ssr-gpu","device":"cpu","query_input_count":0,"num_domains":n,"missing_keys":list(missing),"unexpected_keys":list(unexpected),"tx_logits_shape":list(o["tx_logits"].shape),"z_id_shape":list(o["z_id"].shape),"finite":bool(torch.isfinite(o["tx_logits"]).all() and torch.isfinite(o["z_id"]).all())}
assert not missing and not unexpected and r["finite"] and r["query_input_count"]==0
print(json.dumps(r,ensure_ascii=False,sort_keys=True))
'@ | python -
```

结果：

```json
{"checkpoint":"E:\\type10-7\\automation_reports\\CV-SincNet\\qknnv42_strict_dual125_20260714_183556\\artifacts\\best_joint_safe_ssdg.pth","checkpoint_sha256":"2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98","device":"cpu","finite":true,"missing_keys":[],"num_domains":14,"python_environment":"ssr-gpu","query_input_count":0,"tx_logits_shape":[2,6],"unexpected_keys":[],"z_id_shape":[2,160]}
```

## N607发布计划

代码门已达到`P0=0,P1=0`，但本报告当前仍不构成服务器启动授权。以下字段须在metadata-only提交复核和实时preflight后冻结：

|字段|当前值|
|---|---|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`；direct preflight确认可见|
|Python环境|缺少`ssr-gpu`；仅发现`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，未冒充、未改契约|
|WiSig数据|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；2,359,341,461字节；SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|
|run root|`runs/cvs_full_ablation_phase1_t1_20260728_v1`|
|log root|`logs/cvs_full_ablation_phase1_t1_20260728_v1`|
|主PID/GPU|尚未启动|
|精确启动命令|尚未冻结；发布前写入|

计划同步目标均位于远端项目根的同名相对路径，只同步本次Git提交包含的精确文件；同步后逐文件核对SHA256并执行远端`py_compile`和runner dry-run。

2026-07-28实时preflight证据：普通账号直连成功；8张RTX3090均空闲且无compute app；release/run/log目标均不存在。由于远端没有经审查要求的`ssr-gpu`，runner在SCP、目录创建、seal和launch之前停止。未改变N607状态，结束后本地`ssh.exe`与N607/bridge TCP22残留均为0。继续需要用户明确授权创建/安装N607端`ssr-gpu`，或另走“改用既有`CVS-RFFI`环境”的本地代码变更与独立复审。

## 健康门与停止规则

- 启动后立即核对主PID、CWD、cmdline、run root、16槽和GPU映射。
- 第一批完成/失败row及首个worker wave后，记录launched/completed/succeeded/failed、活跃PID、GPU利用率/显存和异常指纹。
- 任一P0协议/安全违例，或两个不同row在产生有效终局artifact前出现同一规范化异常指纹，停止继续派发并只终止本run已证明归属的进程组。
- 绝不因accuracy、H、floor或其他中间性能值停止。
- 技术早停状态固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留全部部分artifact。

## 预期输出

每个row至少应生成：

- `phase1_ablation_manifest.json`；
- `phase1_terminal_status.json`；
- `phase1_training_completion_receipt.json`；
- `phase1_resource_summary.json`，含wall time、参数量和峰值CUDA显存；
- source-validation-selected checkpoint和source-only prototype；
- 后续W2独立封存的deployment bundle与不可变W2收据；
- per-epoch CSV/JSONL、完整stdout和exit status；
- `ablation_id/git_commit/config_hash/split/seed/parameter_count`；
- source-only UDU、hard-TX、source-side LEO分层指标与资源数据。

## 完成后结果表

当前尚无运行结果。完成后必须按同一arm×seed行填写，不拼接不同row的极值：

|arm|seed|split|参数量|source UDU|hard-TX|LEO clear|LEO low|LEO rain|domain probe|identity leakage|best/final gap|状态|判定|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|待运行|—|`0.07/0.63/0.30`|—|—|—|—|—|—|—|—|—|`NOT_LAUNCHED`|无性能结论|

## 风险与后续检查

1. 双进程/GPU用于吞吐调度，不作为无竞争时延或峰值资源主张；正式资源表另做单进程隔离测量。
2. `P1-A0`参数量精确相等、梯度和真实checkpoint no-query smoke均已验证；W2 bundle smoke仍须在每个arm训练完成后闭合。
3. 每个Phase1 arm必须生成自己的bundle；不得把`P1-FULL`地面状态借给其他arm。
4. 30次训练完成后才按source validation规则封存进入Phase2的`P1-FULL` bundle；target结果不得参与选择。
