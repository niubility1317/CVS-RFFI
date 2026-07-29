# CVS全量消融Phase1标签率v2发布报告

## 实验登记

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_label_20260729_v2`|
|时间|2026-07-29|
|操作者|Codex主代理；N607唯一发布runner由独立子代理负责|
|目标|修复v1的prototype校准导出故障后，完成0.005、0.01、0.02、0.05标签率14个训练行；0.10继续复用Phase1 T1中的5个`P1-FULL`完整行|
|比较对象|同一`P1-FULL`配置，仅改变`f_L/f_U`，固定`f_V=0.30`|
|环境|本地`ssr-gpu`；远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|当前状态|`RUNNING / FIRST_WAVE_HEALTHY`|

## 与v1的关系

v1首行完成200epochs后在prototype导出阶段退出，真实错误为class 0正确source-validation样本不足；prototype文件尚未写出。v1保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不覆盖、不续写、不将其checkpoint直接计作完整行。

v2使用新run/log/release路径，重新执行v1首个失败行和13个未启动行。已有其他批次的完整结果按行身份复用；不同启动批次的数据可以不同，不做跨批次数据一致性检查或数据hash对齐。

## 冻结矩阵

|rho|新训练seed数|seed|
|---:|---:|---|
|0.005|3|7281101–7281103|
|0.010|5|7281101–7281105|
|0.020|3|7281101–7281103|
|0.050|3|7281101–7281103|
|0.100|0|复用T1 `P1-FULL`的7281101–7281105|

合计14个新训练行，每行200epochs。调度上限为GPU0–GPU7每卡2个训练进程；runner根据整机已有进程动态等待空槽，不停止或覆盖Phase1 T1。

## 本地修复与验证

|文件|修复|
|---|---|
|`code/cvsrffi/phase2_prototypes.py`|正确样本不足时使用同一真类全部有限energy的source-validation样本统一回退，并记录correct/all-true计数和校准来源；全部真类样本仍不足时fail closed|
|`code/scripts/run_full_ablation_phase1_t1.py`|先核对terminal和真实退出码；`FAILED_EXPORT`归为技术失败，只有`COMPLETE`却缺失或错绑prototype才判P0|
|`code/tests/test_phase2_prototype_fusion_export.py`|覆盖统一回退成功和真类证据仍不足的拒绝|
|`tests/test_run_full_ablation_phase1_t1.py`|覆盖`FAILED_EXPORT`先于prototype完整性分类|

跨Phase1/Phase2独立回归结果为244 passed、2 skipped、0 failed；实现审查P0=0、P1=0。准确发布提交由sealed plan、review artifact和runner handoff记录，本报告不采用自指Git哈希。

## N607预留位置

|用途|路径|
|---|---|
|release root|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_label_20260729_v2_ae1f9aab`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_label_20260729_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v2`|
|launch log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_label_20260729_v2.launch.out`|

正式命令使用封存plan、不可覆盖run/log/launch路径和远端`CVS-RFFI`Python。发布前只检查当前GPU进程数、精确目标不存在、release checkout干净、入口可编译和sealed dry-run计数；不重审数据集。

## N607发布与首波健康证据

2026-07-29 21:57–22:06 CST完成直连预检、精确目标检查、发布和首波健康核验。T1 runner PID`711523`始终保持运行且未被干预。发布前v2的release、run、log、launch log和PID目标均不存在；只同步bundle、sealed plan和independent review三件发布物。

|检查项|证据|
|---|---|
|Git发布|commit`ae1f9aab1c6095fb5f941d4cebb1cc171100f7a1`；detached HEAD；tracked/untracked clean|
|bundle SHA256|`e3563b774945c9fe2f8e48c0fd8afe7d5ca2b6ddbd041c9c76939b9f68a783d3`|
|sealed plan SHA256|`379cee37f5a36f2a697485f03a234ed7983907547a4510e62d9160e26e5c4f60`|
|independent review SHA256|`cbe6724c2101c1a730d615fd587536f59f61711c86110b431e3d8995d68e0644`|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；PyTorch`2.1.0+cu121`；CUDA可用；8张GPU可见|
|入口编译|runner、`train_ssdg.py`和prototype导出入口通过`py_compile`|
|sealed dry-run|14 rows、14 dispatches、0 direct reuse、0 reexport、14 new trains、14 slots；GPU静态分布`2/2/2/2/2/2/1/1`|
|启动进程|launch PID`823096`；实际runner PID`823097`；CWD、release、plan、run root和log root绑定一致|
|首批实际训练|6行：GPU2两行、GPU5两行、GPU6一行、GPU7一行；其余8行由capacity-aware runner等待空槽|
|整机GPU上限|启动后计算进程数`[2,2,2,2,2,2,1,1]`，所有GPU均不超过2|
|首波进度|截至22:08 CST，6行分别推进至E7–E12/200；runner和6个直接训练子进程均存活|
|闭环artifact计数|status=0、terminal=0、prototype PT=0、prototype JSON=0、completion receipt=0；当前均处于训练中，尚未到导出阶段|
|异常检查|无Traceback、RuntimeError、OOM、参数错误或确定性异常指纹|
|本地连接清理|启动SSH通道因后台bash持有而超时；仅终止精确本地`ssh.exe` PID`5972`，远端任务已landed且持续运行；随后本地`ssh.exe=0`、`ESTABLISHED TCP22=0`，未重复启动|

日志中的`nan`仅出现在无样本分母的诊断字段，例如`overall_tx=nan% (0/0)`；训练loss项保持有限值，不属于系统性技术故障，也未据此进行性能判断或止损。launch log当前为空是因为runner将进度写入每行独立`.out`；6个活动日志已增长至约46–75KB，8个排队日志保持0字节符合capacity-aware等待状态。异常/P0/非零退出指纹计数均为0。

## N607持续监控快照

### 2026-07-29 22:11 CST

|项目|状态|
|---|---|
|runner|PID`823097`存活，release CWD和run绑定未变|
|活动训练|原首波6个直接训练子进程全部存活；GPU2、GPU5各2行，GPU6、GPU7各1行|
|队列|8行等待容量；status=0、complete=0、success=0、fail=0|
|日志增长|6个活动日志约81–138KB，推进至E13–E23/200；8个排队日志仍为0字节|
|闭环artifact|checkpoint=6、prototype PT=0、prototype JSON=0、resource=0、heldout=0、terminal=0、receipt=0|
|技术异常|Traceback、RuntimeError、OOM、参数错误、P0和重复确定性异常指纹均为0|
|整机GPU计算进程|`[2,2,2,2,2,2,1,1]`，没有GPU超过2|
|SSH清理|监控命令结束后本地`ssh.exe=0`、`ESTABLISHED TCP22=0`|

当前6个checkpoint仅是训练中的中间产物；由于terminal、prototype、resource、heldout和receipt均未形成，不计作完成行，也不作性能读取或判断。

### 2026-07-29 22:25 CST

|项目|状态|
|---|---|
|runner|PID`823097`存活，父PID`823096`、release CWD和run绑定未变|
|活动/队列|6个首波直接训练子进程全部存活，8行等待容量|
|状态计数|status=0、complete=0、success=0、fail=0|
|日志增长|6个活动日志约184–363KB，推进至E31–E62/200；8个排队日志仍为0字节|
|闭环artifact|checkpoint=6、prototype PT=0、prototype JSON=0、resource=0、heldout=0、terminal=0、receipt=0|
|技术异常|异常/P0/非零退出/重复确定性指纹均为0|
|整机GPU计算进程|`[2,2,2,2,2,2,1,1]`，所有GPU均不超过2|
|SSH清理|监控命令结束后本地`ssh.exe=0`、`ESTABLISHED TCP22=0`|

尚无首行完成或失败，未触发terminal、receipt和prototype的逐行终态核验。监控未读取任何性能字段，未修改、重启或重新对齐数据。

### 2026-07-29 22:54 CST

|项目|状态|
|---|---|
|runner|PID`823097`存活，父PID`823096`、release CWD和run绑定未变|
|活动/队列|T1释放部分槽位后，capacity-aware runner已将活动训练增至11行；剩余3行等待容量|
|状态计数|status=0、complete=0、success=0、fail=0|
|日志增长|11个活动日志约115–814KB，推进至E19–E140/200；3个排队日志仍为0字节|
|闭环artifact|checkpoint=11、prototype PT=0、prototype JSON=0、resource=0、heldout=0、terminal=0、receipt=0|
|技术异常|异常/P0/非零退出/重复确定性指纹均为0|
|整机GPU计算进程|`[2,2,2,2,2,2,1,1]`，所有GPU均不超过2|
|SSH清理|监控命令结束后本地`ssh.exe=0`、`ESTABLISHED TCP22=0`|

11个checkpoint仍是训练中间产物；没有terminal和完整导出闭环，完成行仍为0。本次监控未读取性能、未干预进程、未重启或对齐数据。

### 2026-07-29 23:11 CST

|项目|状态|
|---|---|
|runner与release|runner PID`823097`、PPID`823096`存活；CWD/cmdline继续绑定release、plan、run/log root、CVS-RFFI Python及`--execute`；release为tracked-clean，HEAD=`ae1f9aab1c6095fb5f941d4cebb1cc171100f7a1`|
|dispatch计数|launched=11、completed=0、succeeded=0、failed=0、nonzero=0、active=11、queued=3|
|子进程绑定|11个活跃训练主进程全部由runner PID`823097`直接持有，CWD、run ID和冻结训练脚本绑定均通过|
|GPU分布|label v2在GPU0–7为`0/1/2/2/2/2/1/1`；与T1合并后的全机训练实验数为`2/2/2/2/2/2/1/1`，所有GPU均不超过2个|
|训练进度与日志|11个活跃行最新`[EPOCH-BEGIN]`为E39–E184；仅作非停滞健康证据。逐行日志总量为6,445,407字节，11个活跃日志均持续增长，3个排队行尚未启动|
|首个/最新完成artifact|status=0，完成行仍为0；当前非空checkpoint=11，但prototype PT=0、prototype JSON=0、resource=0、heldout=0、terminal=0、completion receipt=0，因此这些checkpoint均为中间产物，不能计作完整行|
|技术异常|硬错误、P0、非零退出、异常指纹和重复确定性异常指纹均为0；`runner_summary.json`尚不存在|
|数据边界|未读取性能、未干预或重启任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|全部短连接退出后本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

### 2026-07-29 23:36 CST

|项目|状态|
|---|---|
|本地直连preflight|2026-07-29 23:35:48+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 23:36:27+08:00|
|runner与release|runner PID`823097`、PPID`823096`存活；CWD/cmdline继续绑定release、plan、run/log root、CVS-RFFI Python及`--execute`；release为tracked-clean，HEAD=`ae1f9aab1c6095fb5f941d4cebb1cc171100f7a1`|
|dispatch计数|launched=11、completed=3、succeeded=3、failed=0、nonzero=0、active=8、queued=3；相较23:11首次新增3个完整完成行，没有新增派发行|
|新增完成行|`P1-LABEL-RHO005__train_seed_7281103`、`P1-LABEL-RHO010__train_seed_7281104`、`P1-LABEL-RHO010__train_seed_7281105`；runner记录的真实return code均为0、completion receipt均有效、P0均为false|
|严格独立验收|在冻结release的`PYTHONPATH`和CVS-RFFI Python下，对3行逐一检查7类非空必需artifact，加载checkpoint与prototype PT，解析prototype JSON、resource、heldout、terminal和completion receipt，并核对本行绝对路径、全部嵌入SHA256、COMPLETE状态、退出码及P0机制标志；结果3/3通过、0失败|
|活动/队列|8个活跃训练主进程全部由runner PID`823097`直接持有且绑定通过：rho=0.005种子7281102在GPU1；rho=0.010种子7281101/02/03在GPU3/4/5；rho=0.020种子7281103在GPU2；rho=0.050种子7281101/02/03在GPU3/4/5。其余3行继续等待容量|
|GPU分布|label v2在GPU0–7为`0/1/1/2/2/2/0/0`；与T1合并后的全机训练实验数为`2/2/1/2/2/2/0/0`，所有GPU均不超过2个|
|GPU利用率|GPU0–7依次为96%、98%、0%、95%、97%、98%、0%、0%；对应显存6075、6291、3160、6185、6315、6325、4、4MiB|
|训练进度与日志|8个活跃行最新`[EPOCH-BEGIN]`为E68–E187；仅作非停滞健康证据。逐行日志总量由23:11的6,445,407字节增至8,845,852字节，增加2,400,445字节|
|技术异常|硬错误、P0、非零退出、异常指纹和重复确定性异常指纹均为0；`runner_summary.json`尚不存在|
|数据边界|未读取性能、未干预或重启任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|全部短连接和artifact验收连接退出后本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

#### 23:36新增完整训练行artifact验收

|行|checkpoint/B|prototype PT/B|prototype JSON/B|resource/B|heldout/B|terminal/B|completion receipt/B|return code|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`P1-LABEL-RHO005__train_seed_7281103`|15,263,436|423,939|2,228,880|540|8,138|27,372|5,425|0|完整，通过|
|`P1-LABEL-RHO010__train_seed_7281104`|15,263,372|421,037|2,054,174|540|8,155|27,378|5,404|0|完整，通过|
|`P1-LABEL-RHO010__train_seed_7281105`|15,263,372|423,875|1,975,394|540|8,112|27,336|5,404|0|完整，通过|

## 完整性与止损

每个成功行必须同时具有可加载checkpoint、prototype PT+JSON、resource summary、held-out eval、terminal、completion receipt，并且terminal、receipt与真实退出码都为0。启动或只有checkpoint不算完成。

P0协议/覆盖风险立即停止本run的后续派发；至少两个不同执行行在产生prototype前出现相同确定性异常指纹时停止本run。只终止已证明属于本run的进程树，保留全部部分产物；中间性能不得触发停止。

## 完成后结果表

|rho|seed|labeled数|unlabeled数|source validation数|best epoch|strict UDU|receiver floor|pseudo precision|pseudo coverage|P1-SUP同seed增益|checkpoint|状态|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
