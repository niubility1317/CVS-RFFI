# D110-SCPM＋US-qKNN新source-held G1四臂报告

状态：`ANALYZED / REJECT_D110_SCPM_USQKNN / NO_TARGET25_PROMOTION`

## 1.身份与目标

- run ID：`d110_g1_sourceheld_usqknn_20260802_040736_r1`
- 日期：2026-08-02
- operator：主agent冻结与分析；Terra Max分别实现DA×HEAD核心和新split；后续唯一Terra Max runner负责N607。
- objective：在未打开的新source-held split上完整运行`M0/M_DA/M_HEAD/M_JOINT`63行，分析DA、HEAD、交互、old/new平衡、floor和negative tail。
- 性能边界：新split完成全部预测封存前不得开启truth；历史D104 split只允许机械回归，不能用于D110晋级。

## 2.冻结四臂

|arm|距离|qKNN尺度|
|---|---|---|
|M0|identity平方距离|现有class-specific Student-t尺度|
|M_DA|D110 safe-relative SCPM Mahalanobis距离|同一class-specific公式|
|M_HEAD|identity平方距离|全注册类shared尺度|
|M_JOINT|同一SCPM距离|同一shared尺度公式|

- 四臂共用INT8 support decode、`nu/d_eff/gamma/h0/lambda/clip`、Student-t核与`logsumexp-minus-log-K`。
- shared K>1先求每类无序support pair均值，再做等类平均与原shrink/clip；K1严格取`h0`，所以`M_HEAD=M0`、`M_JOINT=M_DA`是预期可辨识边界。
- SCPM只使用`safe_relative_variances`；公共`1+1/K`不进入qKNN距离。
- query零fit、零update、零selection；无role、quota、truth或batch统计。

## 3.真实G0前置证据

`d110_scpm_g0_oneshot_20260802_033047_r1`已在真实588条tap上完成：K1/K5/K10的argmax变化分别为23/40/96，`zero_changed=[]`，资源预算未超，结论为`G0_PASS_PROCEED_G1 / NO_PERFORMANCE_RESULT`。

## 4.新source-held split预登记

- split ID：`d110_source_seed110813_v1`
- salt：`D110-SCPM-USQKNN|source-held|110813|v1`
- source pool：8400条既有单观测source feature；不重新forward。
- 排除：D103历史query 2478条、D104已打开held 2520条、D110 Phase1 L_s 588条；按physical ID取union。
- 容量：排除后168个receiver×TX×day cell最小合法容量为7，因此固定每cell 7条，共1176条；42个receiver×TX组各28条，四天等权。
- G1结构：7receiver×K1/K5/K10一般行21个，加7receiver×6 held-class×K1行42个，共63行×4臂=252预测单元。
- 每类query数：K1=27、K5=23、K10=18。

### 4.1冻结本地输入

|输入|SHA256|
|---|---|
|D103 8400 dual archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|D105 8400 strict pre-ReLU tap|`6626afbf5d5987b2944b53f9b4bddbb6c9397f4c577accb95cea5e0039b24578`|
|D103历史query manifest|`3fd07b7afcb53b12a08df1643efae80c52917c893cc7453104e68932dc1f5b26`|
|D104 no-truth package manifest|`55780d103e2cb19b4446ef44033446182beac992b42c59753b6910b0910e6710`|
|D104 L_s 588 ID source|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|

builder已证明D103/D105 physical ID同序、`ReLU(pre_relu)=z_id`逐值相同、`z_dom`逐值相同；只输出scorer-only archive、manifest和不含truth值的selection receipt，输出目录不可覆盖。

### 4.2真实split封存结果

状态：`D110_SOURCE_HELD_SPLIT_COMPLETE / UNOPENED_FOR_SCORING / NO_PERFORMANCE_RESULT`

|项目|结果|
|---|---|
|row/cell/group|1176／168／42；每cell 7、每receiver×TX 28|
|排除union|5335个physical ID|
|合法剩余容量|min=7、max=29|
|与D103历史query交集|0|
|与D104旧held交集|0|
|与D110 L_s交集|0|
|selection receipt内部root|`0dd909f39ede47c9ec43e57d67be05a74f54838898880bb16ef5c0264511d852`|
|selection receipt文件SHA|`a1d6c60150e068bd8342dfabf6e74b45f1d41769ea0fa84508e6213817ff3539`|
|scorer manifest SHA|`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`|
|scorer archive SHA|`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`|
|selection receipt含truth值|false|
|performance computed|false|

本地封存目录：`input/d110_sourceheld_split/`。截至本节写入时未调用score、未读取新split性能、未生成任何预测。

## 5.本地实现与验证

|部分|commit|证据|
|---|---|---|
|D110理论与G0结论|`1e6b94d3`|真实G0三K非零|
|US-qKNN四臂core|`d9f8b62c`|M0 bit-match；D110专项14项通过|
|新split builder|`48a41e3b`|新旧split回归13项通过；真实split已封存且未评分|
|G1一次性入口|`ef8d3109`|7项目标测试、`py_compile`、`git diff --check`通过|

独立发布复审：`P0=0 / P1=0 / P2=0 / GO`。复审确认prediction manifest冻结truth seal SHA，score在创建truth-open event前校验prediction→package manifest→truth seal链；替换seal的负测在事件生成前fail-closed。该结论只表示发布就绪，不是性能结果。

## 6.不可变发布包

|artifact|SHA256|
|---|---|
|`release/d110_code_ef8d3109.zip`|`2ad3af6dea23c6c324a212dc7e7d19fcb900937bcdca82b0cfcd62c8edd2f28b`|
|`release/d110_sourceheld_split.zip`|`9083ddfd5abf73272c1ccd53dc7a4853321dbfb7e60a4dc166277f8a3c71ecff`|
|G1入口Python|`5f693a8438bac789b3759c2adc097aa95136256219851b7bc97d938c8c9beb71`|
|G1入口测试|`3dcd59a42dfdc2c68078252ad27b07028987edb263a61b64d4546c126fc796f7`|

## 7.N607预登记

- remote root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_g1_sourceheld_usqknn_20260802_040736_r1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 运行性质：冻结NumPy特征推断，不训练；唯一runner仍需preflight和资源记录。
- 同步映射：两个本地`release/*.zip`同步到remote root，分别解压到`release_code/`与`input/`；不改动服务器已有D106 tap。
- 固定D106 tap：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`，SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`。
- 唯一子进程依次执行`prepare`→`predict`→`score`；`predict`必须先封存63行、252个arm-row单元，`score`才能打开truth。
- 输出：`output/packages/package_manifest.json`、`output/predictions/prediction_manifest.json`、`output/truth_open_event.json`、`output/held_scores.json`、`logs/runner.log`、`launch.pid`、`launch.exit`。
- fresh-run retry：未授权；技术失败保留r1并返回本地修具体缺陷。
- 停止条件：仅P0协议/覆盖/错误SHA、确定性异常或零prediction；不得按性能停止。

后续只补：N607实际命令、PID/GPU/进程与日志证据、完整同row结果及推荐。

## 8.唯一runner启动记录

- runner：`/root/d110_n607_runner`；本run仅由该runner登陆、落地、启动、监控和回收，主agent不并发操作同一run ID。
- 本地复核：代码工作树HEAD为`9e04a969867949660df40445615966be67821ad2`，冻结G1代码commit为`ef8d3109a3c5135c2b0a08ba9b221beaecdfb3aa`；两份release zip的SHA256与第6节逐字匹配。
- 发布状态：`LOCAL_VERIFIED`；下一步为只读direct preflight、GPU/活跃任务记录和远端不可覆盖root检查。通过后才创建root并同步两份zip。
- 预登记运行链：在remote root内使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`依次执行`prepare`、`predict`、`score`；预测封存不足63行或252个arm-row单元时不允许进入`score`。
- 健康与停止：只检查SHA、覆盖、PID/CWD/cmdline绑定、日志增长和确定性异常；不依据任何中间性能指标停止；fresh retry未授权。
- 版本承载说明：本报告目录不属于Git工作树；按本次唯一runner授权直接更新本地报告，不改动Git镜像。

### 8.1直连预检与资源快照

- 结果：`PREFLIGHT_OK`。直连`N607`使用普通账号`szu2070436088`成功；N607时间为2026-08-02 04:41:53 CST，项目根可见。
- 目标root检查：`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_g1_sourceheld_usqknn_20260802_040736_r1`在创建前不存在。
- 资源：GPU0--GPU7均为RTX3090，利用率0%、显存1MiB；`nvidia-smi`未列出compute process。仅见系统`unattended-upgrade-shutdown`Python进程，不属于实验；不作干预。
- 远端依赖：固定Python可执行且版本为Python3.10.19；固定D106 tap存在且SHA256为`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`，与预登记一致。
- 容量：项目所在`/home`卷可用7.4T。预检短连接结束后本机未残留`ssh.exe`或到`172.31.111.215:22`的TCP连接。

### 8.2不可覆盖落地

- 状态转换：`LOCAL_VERIFIED -> LANDED`。远端以`mkdir`创建预登记root，创建前不存在；目录模式为775。未触及其他run、tap、数据集或检查点。
- 仅同步：`d110_code_ef8d3109.zip`与`d110_sourceheld_split.zip`。远端SHA256分别为`2ad3af6dea23c6c324a212dc7e7d19fcb900937bcdca82b0cfcd62c8edd2f28b`和`9083ddfd5abf73272c1ccd53dc7a4853321dbfb7e60a4dc166277f8a3c71ecff`，均与本地冻结值一致。
- 解压位置：代码到`release_code/`，split到`input/d110_sourceheld_split/`，并创建空的`output/`和`logs/`。入口脚本远端SHA256为`5f693a8438bac789b3759c2adc097aa95136256219851b7bc97d938c8c9beb71`，固定Python的`py_compile`通过；三个预登记输入文件均存在。
- 包内容说明：release code zip不含测试文件，远端仅执行预登记的入口`py_compile`；本地目标测试已在冻结阶段通过。未补传测试或任何额外文件。
- 落地后无D110子进程、预测覆盖为0/63、arm-row覆盖为0/252、异常指纹为无；每次短连接后本机SSH进程和到N607的TCP22连接均为空。

### 8.3冻结启动命令与首波检查

远端CWD固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/d110_g1_sourceheld_usqknn_20260802_040736_r1`，由一个`nohup`子进程顺序运行：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python release_code/code/scripts/run_d110_g1_sourceheld_one_shot.py prepare --source-val-archive input/d110_sourceheld_split/scorer_only/source_val/features.npz --source-val-manifest input/d110_sourceheld_split/scorer_only/source_val/manifest.json --output-dir output/packages
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python release_code/code/scripts/run_d110_g1_sourceheld_one_shot.py predict --package-root output/packages --d106-tap-archive /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --run-id d110_g1_sourceheld_usqknn_20260802_040736_r1 --output-dir output/predictions
# 仅在prediction manifest证明row_count=63、arm_row_prediction_unit_count=252且63个row receipt均唯一后：
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python release_code/code/scripts/run_d110_g1_sourceheld_one_shot.py score --prediction-root output/predictions --truth-json output/packages/scorer_only/truth.json --truth-input-seal-json output/packages/scorer_only/truth_input_seal.json --truth-open-event-json output/truth_open_event.json --output-json output/held_scores.json
```

- detached记录：`logs/runner.log`、`launch.pid`、`launch.exit`；启动后立即检查PID、CWD、cmdline、run-root、日志增长、行文件数、GPU和异常指纹。
- 首个row与首个wave只用于技术健康：检查预测文件增长、无traceback/错误SHA/覆盖故障；不读取或依据任何性能值做停止或选择。

### 8.4运行完成、回收与清理

- 状态转换：`LANDED -> RUNNING -> ARTIFACTS_COMPLETE`。唯一launch PID为`3686232`，其`launch.exit=0`且已退出；未观察到run-owned进程残留。
- 完整顺序：`runner.log`记录`prepare -> predict -> prediction_closure_validate -> score -> RUNNER_COMPLETE`；封存检查在打开truth前明确记录`rows=63`、`arm_rows=252`。
- 覆盖：预测目录有63个row文件；prediction manifest声明63行、252个arm-row单元、63个唯一prediction receipt；独立score artifact含63个performance row。此处仅记录闭环，不在runner阶段读取或解释性能数值。
- 完成artifact：`package_manifest.json`、truth seal与truth、63行预测及manifest、truth-open event、held score、runner日志、PID/exit与source-held selection receipt均已回收到[retrieved](retrieved/)；逐文件SHA和coverage receipt见[ARTIFACT_INTEGRITY.md](retrieved/ARTIFACT_INTEGRITY.md)。
- 异常与安全：runner日志无Traceback/异常指纹；最终GPU0--GPU7均为0%利用率、1MiB显存，launch PID已退出。本机每次短连接和SCP后均未残留`ssh.exe`或N607 TCP22连接。一次监控查询曾因本地相对路径读取失败提前结束，但未写入远端、未影响runner，后续以绝对路径复核完成。
- 后续：保持remote root和本地回收件不覆盖、不删除；主agent应使用同一run的完整同row artifact进入`ANALYZED`并作性能结论与下一候选决策。

## 9.完整同row性能分析

分析输入为本run回收的`retrieved/held_scores.json`，SHA256=`81dd9315c3c81a58cf7da43154b2e3d4a60f3750e4bf568d02f4ecb3d41be9e4`。全63行一次性读取，没有挑选receiver、K或held class。

### 9.1全类一般行（21行）

|K|arm|机制|balanced accuracy|all-class floor|correct/query|
|---:|---|---|---:|---:|---:|
|1|M0|identity＋class-specific|84.0388%|57.6720%|953/1134|
|1|M_DA|SCPM＋class-specific|82.4515%|44.4444%|935/1134|
|1|M_HEAD|identity＋shared|84.0388%|57.6720%|953/1134|
|1|M_JOINT|SCPM＋shared|82.4515%|44.4444%|935/1134|
|5|M0|identity＋class-specific|84.9896%|60.8696%|821/966|
|5|M_DA|SCPM＋class-specific|84.2650%|59.6273%|814/966|
|5|M_HEAD|identity＋shared|84.9896%|59.0062%|821/966|
|5|M_JOINT|SCPM＋shared|84.5756%|59.0062%|817/966|
|10|M0|identity＋class-specific|84.3915%|54.7619%|638/756|
|10|M_DA|SCPM＋class-specific|84.2593%|56.3492%|637/756|
|10|M_HEAD|identity＋shared|84.5238%|56.3492%|639/756|
|10|M_JOINT|SCPM＋shared|84.3915%|57.1429%|638/756|

### 9.2K1登记行（42行）

|arm|机制|old BA|seen-new accuracy|H old/new|all-class floor|correct/query|
|---|---|---:|---:|---:|---:|---:|
|M0|identity＋class-specific|84.0388%|84.0388%|82.3063%|57.6720%|5718/6804|
|M_DA|SCPM＋class-specific|82.4515%|82.4515%|79.5106%|44.4444%|5610/6804|
|M_HEAD|identity＋shared|84.0388%|84.0388%|82.3063%|57.6720%|5718/6804|
|M_JOINT|SCPM＋shared|82.4515%|82.4515%|79.5106%|44.4444%|5610/6804|

K1按冻结设计严格满足`M_HEAD=M0`、`M_JOINT=M_DA`，42个登记行逐行度量完全相同，因此K1不能为shared head提供收益证据。

### 9.3简单效应、交互和稳定性

|effect|范围|主指标均值变化|正/零/负行|结论|
|---|---|---:|---:|---|
|DA_AT_BASE|全42个K1登记行|H -2.7957pp；old BA -1.5873pp；new -1.5873pp；old floor -10.0529pp|H 12/0/30|明显负收益|
|JOINT_VS_M0|全42个K1登记行|H -2.7957pp；all-class floor -13.2275pp；correct -108|H 12/0/30|不晋级|
|JOINT_VS_M0|21个一般行|BA -0.6671pp；floor -4.2366pp|BA 7/1/13|负向且不稳定|
|HEAD_AT_ID|21个一般行|BA +0.0441pp；floor -0.0920pp|BA 5/11/5|数值近乎恒等，无意义正收益|
|HEAD_AT_DA|21个一般行|BA +0.1476pp；floor +0.0575pp|BA 4/16/1|稀疏微小，无法抵消DA退化|
|交互效应|21个一般行|BA +0.1035pp；floor +0.1495pp|BA 5/12/4|稀疏、非稳定|

receiver稳定性进一步否定SCPM：K1登记的`JOINT_VS_M0`在7个receiver中5个H下降；`2-1`的平均H下降19.9457pp、old floor下降61.7284pp。虽然`18-2`平均H上升5.0217pp，但这是局部正向，不能抵消整体退化，也不得据此挑receiver晋级。

### 9.4机制解释与决策

SCPM把Phase1条件方差解释为“域扰动强度”并对角逆加权，但实证表明这个代理不具有稳定的判别不变性：低方差维度可能同时是receiver特异且会被放大的方向，所以发生大幅floor崩落。shared head只改变尺度，K1时理论上恒等，K5/K10也只有近零、稀疏变化，说明它没有解决分类边界问题。

最终决策：`REJECT_D110_SCPM_USQKNN / NO_TARGET25_PROMOTION`。D110没有出现可称为方法级正收益的版本；仅有局部正行和近零head变化。不调SCPM cap、不扫描权重、不运行Target25，直接转入下一个从理论构造的轻量方法。下一方法必须显式处理K1可识别性，且不再使用“方差小=应放大”的对角度量假设。

