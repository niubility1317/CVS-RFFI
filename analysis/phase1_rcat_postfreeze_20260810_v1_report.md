# Phase1 P1-RCAT后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_rcat_postfreeze_20260810_v1`
- 日期：2026-08-10
- 操作角色：主代理冻结评价合同与判定门；唯一N607 Runner负责落地、唯一启动、技术监控和小工件回收
- 当前状态：`LOCAL_VERIFIED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`
- 训练输入：`phase1_rcat12_20260810_v1`，已技术闭合12/12臂，训练报告SHA=`f57552874c91e538eafce0da8ff156a2f2936c891770790f7245d4dca70f0879`，Git mirror commit=`f126811bfffa6dd73bcb3d61bd1350fe9b59b9e5`
- 目标：以冻结42步对同fold C/G执行clean、三LEO、fixed400 proxy和连续Gaussian-NLL公平评价，生成6份pair JSON及F6矩阵聚合。
- 边界：技术完成不等于性能通过；任何完整门失败永久`REJECT_P1_RCAT_PERMANENT`，全部通过也只能`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`。

## 2.冻结版本与本地验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 后冻结实现commit：`51bc94b935fa289e0ce624f0157efdddc0a5d00d`
- 独立actual-diff终裁：`P0=0/P1=0/ALLOW`

|文件|工作树SHA256|用途|
|---|---|---|
|`analysis/phase1_rcat_postfreeze_design_20260810.md`|`cff7d149bd166ebf9110c8e97bcd102aedcf254d497812b4c530d4eec7f4c1ba`|后冻结设计与追踪|
|`code/export_phase1_rcat_features.py`|`2d0603d23de51447afbbe532931ef76e4d2a9f34f0d0c5fe54151c2d124d7673`|clean L/V/proxy专用导出|
|`code/export_phase1_rcat_leo_features.py`|`404c51f15f2a872c7253bb6cc7e158471c20a4fe5c06d35d4321e5a57e859cdc`|三LEO导出与物理绑定|
|`code/evaluate_phase1_rcat_postfreeze_pair.py`|`41472ca0e178ecc4806e63a7f26976d873b8b272ca9d413d9de32c513bf78408`|同fold C/G评分与F6聚合|
|`code/tests/test_phase1_rcat_postfreeze.py`|`1681843dbdadea0556ff5065bd7318f143a1f8f18604d8017bf134cbb75e8cc0`|receipt、物理绑定、篡改与门测试|
|`code/scripts/launch_phase1_rcat_postfreeze_20260810.sh`|`dcc709ba84e4f23021c7ad4a0e5ec7d85c1afdfdc96658740801477c6dec5bf1`|冻结42步launcher，Git mode100755|

本地`ssr-gpu`串行验证：

- `py_compile`：通过。
- RCAT后冻结专项：`33 passed`。
- RCRMD共享公平核回归：`27 passed`。
- `bash -n`：通过。
- dry-run：精确42步=`12 clean+12 LEO/binding+12 proxy+6 pair`。
- `git diff --check`：通过。
- 旧身份反例：4个`icmt_*`字段分别注入clean manifest并同步重算proxy JSON/CSV，全部fail-closed。

Runner必须从实现commit生成完整无prefix、LF-only归档，记录archive SHA/大小/member、`code/code=0`、6个目标member SHA与launcher mode；不得把Windows mixed-EOL工作树直接作为release。

## 3.冻结评价合同

- clean专用NPZ角色：`labeled_fit`只来自训练L；`source_validation_known`只作known评分；`proxy_unknown`只作proxy评分。U零forward，V/proxy零fit。
- 只用L的`z_id=feat_joint`拟合float64分段totalized-L2对角Gaussian；精确zero映射0并保留，nonfinite fatal。
- 每类方差`ddof=1`；`0.9×class+0.1×class-equal pooled`收缩；方差下限`1e-6`；完整Gaussian-NLL与稳定logsumexp连续`u`。
- fixed proxy：days=`2021_03_01,2021_03_08`；RX=`1-1,1-19,14-7,18-2,19-2,2-1`；selection seed=`7281148`；max/TX=`400`；total=`400`。
- LEO绑定必须封存ManySig path/SHA、selection、physical ID及每scene TX/RX/day完整覆盖。
- 每个pair重新核C/G checkpoint和RCAT receipt：Rs0..6、divisor28、84格、共同训练binding；C aux N/A/0；G positive q、feat_joint/shared encoder VJP nonzero、head aux None/zero且共同head路径live。
- RCAT manifest不得包含任何旧`icmt_*`身份字段。
- F6逐项重开F1--F5的clean、LEO、binding、proxy JSON/CSV和当前checkpoint，核当前SHA并按冻结函数重算；不得信任prior pair自报summary。

## 4.非补偿判定门

- clean：6/6折的overall、min-class、min-RX、min-day均不低于同fold C−2pp。
- LEO：18/18场景格的四项floor均不低于同fold C−2pp。
- overall：每fold三场景等权overall增量均≥0，且全18格等权overall增量≥0。
- proxy：每fold`ΔAUROC>0`且`Δ(mean u_proxy−mean u_V)>0`，必须6/6。
- floor、overall与proxy互不补偿，不得用平均值覆盖单格失败。

## 5.N607路径、资源与唯一命令

- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 预计release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat_postfreeze_20260810_v1_51bc94b9`
- 精确CWD：`<release>/code`
- 训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcat12_20260810_v1`
- 后冻结根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcat_postfreeze_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat_postfreeze_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat_postfreeze_20260810_v1_launcher.out`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`

GPU映射与训练一致：F1C+F5G/0，F1G+F5C/1，F2C+F6G/2，F2G+F6C/3，F3C/4，F3G/5，F4C/6，F4G/7；pair按F1--F6 CPU串行。GPU7存在独立SCB v5构建时，只读记录并不得干预；RCAT后冻结为第二个实验，仍须遵守资源上限。

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat_postfreeze_20260810_v1_51bc94b9/code && nohup env POSTFREEZE_RUN_ID=phase1_rcat_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat_postfreeze_20260810_v1_51bc94b9/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcat12_20260810_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcat_postfreeze_20260810_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat_postfreeze_20260810_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat_postfreeze_20260810_v1_51bc94b9/code/scripts/launch_phase1_rcat_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &
```

调用只能1次，retry=`NO`。调用端超时必须先清本地SSH/TCP22并只读核run/log/PID/CWD/cmdline是否landed，严禁重发。

## 6.技术健康、停止与预期工件

预期：12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON、12 proxy CSV、6 pair JSON、18阶段日志、PID表和outer。技术门检查schema/root/matrix/training-root/checkpoint/receipt/common/proxy/physical binding及F6 aggregate。

停止仅限错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct candidate在产出目标工件前出现相同确定性异常。停止前精确核run-owned PID/CWD/cmdline，只停止本run并保留partial。不得读取accuracy、floor、AUROC、u-gap或其他性能值决定是否停止。

Runner只回收小JSON/CSV/binding/log/PID/manifest，不下载checkpoint或NPZ；技术交接先标记`PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`。主代理在全部pair和原始工件闭合后才读取性能并作最终判定。

## 7.Runner落地与技术交接

- 当前状态：`PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`。本节只记录版本、落地、资源、工件和技术绑定，不读取或解释accuracy、floor、AUROC、u-gap等性能字段。
- 实现commit：`51bc94b935fa289e0ce624f0157efdddc0a5d00d`；预注册commit：`d77158830a895340cba22bd3677407b9e72ecff2`。
- 归档由实现commit生成，完整无prefix、4939 members、`code/code=0`、文本成员CRLF计数为0；LF归档SHA=`e578fc977ca7f664851820ce65514ba5e151ca234c5490eaa667685ad8ea9776`，大小=`34479529`字节。远端临时归档SHA、大小和成员数一致，解包后release launcher mode=`775`，六个目标member逐项一致：

|member|LF SHA256|
|---|---|
|`analysis/phase1_rcat_postfreeze_design_20260810.md`|`cff7d149bd166ebf9110c8e97bcd102aedcf254d497812b4c530d4eec7f4c1ba`|
|`code/export_phase1_rcat_features.py`|`2d0603d23de51447afbbe532931ef76e4d2a9f34f0d0c5fe54151c2d124d7673`|
|`code/export_phase1_rcat_leo_features.py`|`404c51f15f2a872c7253bb6cc7e158471c20a4fe5c06d35d4321e5a57e859cdc`|
|`code/evaluate_phase1_rcat_postfreeze_pair.py`|`41472ca0e178ecc4806e63a7f26976d873b8b272ca9d413d9de32c513bf78408`|
|`code/tests/test_phase1_rcat_postfreeze.py`|`1681843dbdadea0556ff5065bd7318f143a1f8f18604d8017bf134cbb75e8cc0`|
|`code/scripts/launch_phase1_rcat_postfreeze_20260810.sh`|`dcc709ba84e4f23021c7ad4a0e5ec7d85c1afdfdc96658740801477c6dec5bf1`|

- direct N607 preflight通过；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；训练root的12个`final_ssdg.pth`仅远端逐SHA核验并未下载，12个RCAT receipt schema/root均闭合。启动前release、postfreeze run、log、outer及临时路径均`ABSENT`。GPU7既有SCB实时PID=`958466`（父PID=`958333`，约845MiB）；未干预，GPU0--6启动前空闲。
- 远端静态门全部通过：release内5个Python入口`py_compile`、4个公开CLI`--help`、`bash -n`；冻结环境dry-run精确42步=`12 clean+12 LEO/binding+12 proxy+6 pair`，RCAT candidate/root、固定seed/400和无旧身份均通过，且未留下release pycache。
- 第5节逐字唯一启动命令调用次数=`1`，调用端约77秒无stdout异常返回；按`retry=NO`未重发。只读确认已落地：wrapper PID=`1038841`、launcher PID=`1038842`，12个candidate PID写入`candidate_pids.tsv`并绑定冻结release/code与预登记GPU映射（0/1/2/3/4/5/6/7）；F6 pair短时PID=`1044070`。全部进程随后自然退出；未执行kill，run-owned进程终态=`0`。
- 技术工件闭合：12/12 clean NPZ、12/12 LEO NPZ、12/12 LEO binding、12/12 proxy JSON、12/12 proxy CSV、6/6 pair JSON、18阶段日志和PID表均存在；6份pair JSON schema、matrix/output root、training root、common training binding、receipt revalidation、proxy recomputation技术键及F6 aggregate键均存在；Traceback、RuntimeError、argparse、OOM、SIGSEGV等技术异常文件名为0。上述只作技术闭合记录，不作性能判断。
- 小工件bundle已回收至`automation_reports/CV-SincNet/phase1_rcat_postfreeze_20260810_v1/artifacts/phase1_rcat_postfreeze_20260810_v1_small_bundle.tar.gz`：SHA=`0d9cdd92bc764eebce1a46d681dea91757ec36ecc039235c02ea7f0be759415a`，大小=`5534914`字节，63 members；manifest SHA=`444e8dea10490e2475880a2b7ae3f6f2ec67b2e81c9130d25c31e0857c9ef8a3`，大小=`13872`字节。bundle含18日志、12 binding、12 proxy JSON、12 proxy CSV、6 pair JSON、PID/outer和manifest，不含`.pth/.npz`；远端临时archive、bundle及bundle目录均已清理并核验`ABSENT`。
- 每次SSH/SCP后本地`ssh.exe=0`、N607 TCP22=`0`；release、run和log按约定保留。Runner不启动后续run，不做性能读取/解释或晋级签字。
