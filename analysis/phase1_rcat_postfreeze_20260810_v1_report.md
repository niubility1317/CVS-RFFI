# Phase1 P1-RCAT后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_rcat_postfreeze_20260810_v1`
- 日期：2026-08-10
- 操作角色：主代理冻结评价合同与判定门；唯一N607 Runner负责落地、唯一启动、技术监控和小工件回收
- 当前状态：`ANALYZED / REJECT_P1_RCAT_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`
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

## 8.主代理同排性能分析

### 8.1分析输入与完整性

- 主代理只在Runner完成42步、6/6技术绑定和小工件回收后读取性能字段。分析输入为回收bundle内6份pair JSON；manifest SHA=`444e8dea10490e2475880a2b7ae3f6f2ec67b2e81c9130d25c31e0857c9ef8a3`。
- 6份pair JSON的本地字节数与SHA均和manifest逐项一致：F1=`06224031edad9f57cf70ba36e209e5fe7ba310dbe2a49ed054683901e70836e6`，F2=`8e3a9fd2e151d2514a216ce1593f7f3462492ba88d8c2871e18be2cb7f98e3a2`，F3=`82fed47d59499ec7bfcc47ca22ffc1f96ceb7fa88cdc1ddb8d38539cd7140b14`，F4=`8a99e5321bafc8ddafad2ef2824d5dffc4fa99e61b53d24dd7b7fa80f922c8d7`，F5=`c2e6ed0057b79ba6772cac6777642a2e84883cc2b6d6da42e0d2e83b417c2b1d`，F6=`23fdf1d49a2c2396273ff581272e9a480365179cb6a183caf3f4d5f60531b20a`。
- 6/6 pair的`technical_binding.passed=true`；F6的matrix aggregate确认F1--F5从当前clean、LEO、binding、proxy JSON/CSV原始工件按当前SHA重开并重算，不能用可改写摘要替代。
- 下表中accuracy绝对值按百分比显示，所有`G-C`均为百分点；proxy AUROC为0--1标度，`u-gap=mean(u_proxy)-mean(u_V)`。

### 8.2clean source-validation

|fold|C overall(%)|G overall(%)|Δoverall(pp)|Δmin-class(pp)|Δmin-RX(pp)|Δmin-day(pp)|四floor|
|---|---:|---:|---:|---:|---:|---:|---|
|F1|99.315476|99.309524|-0.005952|+0.095238|+0.125000|0.000000|PASS|
|F2|99.232143|99.178571|-0.053571|-0.428571|-0.416667|-0.059524|PASS|
|F3|99.345238|99.345238|0.000000|-0.119048|0.000000|+0.035714|PASS|
|F4|99.291667|99.279762|-0.011905|+0.190476|+0.250000|-0.047619|PASS|
|F5|97.994048|98.160714|+0.166667|+2.023810|+1.125000|+0.250000|PASS|
|F6|97.375000|97.773810|+0.398810|+1.000000|+3.166667|+0.166667|PASS|

clean四floor达到`6/6`。RCAT是本轮已完成候选中首个在六折clean上全部守住`C-2pp`地板的机制，但该门只证明没有clean灾难性退化，不能补偿LEO或proxy失败。

### 8.3LEO三场景18格

|fold|场景|C overall(%)|G overall(%)|Δoverall(pp)|Δmin-class(pp)|Δmin-RX(pp)|Δmin-day(pp)|四floor|
|---|---|---:|---:|---:|---:|---:|---:|---|
|F1|clear|96.691176|96.323529|-0.367647|-0.694444|-2.061856|-0.433663|FAIL|
|F1|low-elev|95.772059|95.036765|-0.735294|-1.562500|-1.123596|-1.442308|PASS|
|F1|rain|96.093750|95.117188|-0.976562|-1.562500|-2.531646|-1.339286|FAIL|
|F2|clear|93.014706|94.301471|+1.286765|+4.166667|+2.061856|+1.351351|PASS|
|F2|low-elev|89.522059|90.441176|+0.919118|+8.593750|+3.370787|+0.137363|PASS|
|F2|rain|90.234375|92.187500|+1.953125|+9.375000|+2.531646|+2.728175|PASS|
|F3|clear|94.301471|95.036765|+0.735294|+0.694444|-1.030928|+1.242236|PASS|
|F3|low-elev|87.500000|87.316176|-0.183824|-0.781250|+2.247191|-0.480769|PASS|
|F3|rain|88.281250|89.062500|+0.781250|+0.781250|-1.265823|+1.388889|PASS|
|F4|clear|93.933824|92.095588|-1.838235|-11.805556|-4.123711|-2.078787|FAIL|
|F4|low-elev|90.073529|88.419118|-1.654412|-7.812500|-5.617978|-2.564103|FAIL|
|F4|rain|90.039062|89.843750|-0.195312|-7.031250|+0.566289|-0.694444|FAIL|
|F5|clear|77.389706|80.330882|+2.941176|+3.645833|+10.309278|+3.332214|PASS|
|F5|low-elev|70.036765|71.875000|+1.838235|+4.947917|+2.247191|+0.480769|PASS|
|F5|rain|65.625000|68.359375|+2.734375|+19.531250|+3.947368|+2.182540|PASS|
|F6|clear|80.882353|77.389706|-3.492647|-18.055556|-17.910448|-5.660008|FAIL|
|F6|low-elev|81.617647|82.904412|+1.286765|-13.281250|+5.617978|+2.884615|FAIL|
|F6|rain|81.445312|79.101562|-2.343750|-15.625000|-2.531646|-3.125000|FAIL|

LEO四floor只达到`10/18`，完整fold为`3/6`（F2、F3、F5）。逐fold三场景overall等权差为F1=`-0.693168pp`、F2=`+1.386336pp`、F3=`+0.444240pp`、F4=`-1.229320pp`、F5=`+2.504596pp`、F6=`-1.516544pp`，所以fold overall也仅`3/6`。

全18格等权变化为overall=`+0.149357pp`、min-class=`-1.470872pp`、min-RX=`-0.294336pp`、min-day=`-0.116123pp`。正overall均值主要由F2、F3、F5贡献，不能补偿F1、F4、F6的失败。最严重的非退化问题集中在F4和F6：F4三个场景的min-class均下降超过7pp，F6三个场景的min-class均下降超过13pp，且F6-clear的min-RX下降`17.910448pp`。

### 8.4fixed400 proxy连续双门

|fold|C AUROC|G AUROC|ΔAUROC|C u-gap|G u-gap|Δu-gap|双严格门|
|---|---:|---:|---:|---:|---:|---:|---|
|F1|0.807939|0.792204|-0.015736|1375.302698|1038.972129|-336.330569|FAIL|
|F2|0.673646|0.507782|-0.165864|576.808697|504.664390|-72.144307|FAIL|
|F3|0.917543|0.932568|+0.015025|1619.587309|1717.748103|+98.160794|PASS|
|F4|0.463038|0.469497|+0.006459|2433.861949|1750.444354|-683.417595|FAIL|
|F5|0.875969|0.918047|+0.042078|335.777581|465.032084|+129.254503|PASS|
|F6|0.836185|0.796129|-0.040056|1365.882002|1565.126395|+199.244393|FAIL|

proxy双严格门仅`2/6`（F3、F5）。F4仅AUROC上升而u-gap下降，F6仅u-gap上升而AUROC下降；F1、F2两项同时下降。proxy是后冻结、TX隔离、L-only fit的连续几何诊断，不是真实unknown能力证据，也不能由clean或LEO结果补偿。

### 8.5完整非补偿门与最终裁决

|冻结门|RCAT结果|要求|判定|
|---|---:|---:|---|
|技术绑定|6/6|6/6|PASS|
|clean四floor|6/6 fold|6/6|PASS|
|LEO四floor|10/18 cell；3/6 fold完整|18/18；6/6|FAIL|
|逐fold三场景overall|3/6非负|6/6|FAIL|
|全18格overall|+0.149357pp|≥0|PASS|
|proxy连续双门|2/6 fold|6/6|FAIL|

F6 matrix aggregate的最终verdict为`REJECT_P1_RCAT_PERMANENT`，主代理复核同意。根据预注册，不调`lambda_rcat`、不换fold、receiver、TX、场景或seed，不选择F2/F3/F5局部成功，不以全18格正均值补偿失败，也不重试同一机制。RCAT不进入Phase3候选。

## 9.与前三个已完成候选的同合同复盘

|候选|clean四floor|LEO四floor|完整LEO fold|全18格overall|proxy双门|最终状态|
|---|---:|---:|---:|---:|---:|---|
|ICMT|5/6|3/18|0/6|-4.309002pp|1/6|永久拒绝|
|CAGM|5/6|9/18|1/6|-0.128294pp|4/6|永久拒绝|
|RCRMD|5/6|15/18|5/6|+2.180990pp|0/6|永久拒绝|
|RCAT|6/6|10/18|3/6|+0.149357pp|2/6|永久拒绝|

RCAT验证了“同物理行方向传输”确实比ICMT更少引起整体LEO负迁移，并首次守住全部clean折；但它没有稳定控制F4/F6的class/RX最坏尾部，proxy连续几何也只在两折同向改善。与RCRMD相比，RCAT换来了更好的clean稳定性和部分proxy恢复，却丢失了5个LEO通过格及2个完整fold；与CAGM相比，它改善了LEO整体但弱化了proxy一致性。因此当前证据不支持继续沿用单一逐样本角度对齐作为完整解法，也不支持把任何局部fold成功解释为receiver、day或unknown能力修复。

最终状态：`ANALYZED / REJECT_P1_RCAT_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`。
