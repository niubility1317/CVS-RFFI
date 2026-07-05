# phase1_epoc_r7_floor_protected_shell_20260706

## 基本信息

|字段|内容|
|---|---|
|experiment_id|phase1_epoc_r7_floor_protected_shell_20260706|
|timestamp|2026-07-06 03:55 CST|
|operator|Codex|
|objective|在协同推理、OSPR-CI++、feature geometry、target-old prototype/ridge/MLP上限均未达到目标后，启动更底层source-only地面训练候选，优先保护旧类floor并修复LEO特征几何。|
|base/teacher|`ADV3B02_CORE90_SOFT_E200`，checkpoint:`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|route|`source_only_floor_protected_feature_shell`|
|status|n607_running_monitor_e089_e089|

## 协议边界

- 地面训练只使用`ManySig.pkl`源域旧类数据，`labeled_ratio=0.10`。
- 不加载`ManyTx.pkl`，不传入`--new_wisig_pkl`，不使用真实`Y_unknown`或目标接收机`R_t`样本。
- 所有未知拒识训练信号来自虚拟unknown、source-heldout proxy逻辑、open-world feature shell、source episode半径约束和soft inter-class mixup，不接触真实Stage2 `target_unknown`。
- 输出prototype只用于后续Stage2-C评估；本轮训练完成不等于部署成功。

## 设计动机

上一轮证据显示：

|证据|结论|
|---|---|
|OSPR-CI/qknn8协同|资源满足，但旧类、新类和未知拒识无法同row达标。|
|OSPR-CI++|可把未知拒识推到约99%，但旧类和seen-new几乎崩塌。|
|feature-space upper bound|source-calibrated open boundary约89%-90%未知拒识，未达到99%。|
|target-old prototype/ridge/MLP上限|MLP support train_acc=100%，target query old_acc最高约73%，说明不是线性头不足，而是LEO目标域特征几何不足。|

R7的核心改变：

- 强化teacher clean/sat/z_id蒸馏，避免学生偏离ADV3B02旧类能力。
- 加强`z_id` compactness和source episode三sigma半径约束，优先保护旧类每类floor。
- 降低R5/R6中可能导致已知类崩塌的proxy/soft unknown强度，把open-set shell作为轻约束而非主导目标。
- 继续导出Phase2 prototype，后续必须回到Stage2-C qknn8+协同推理评估。

## 候选矩阵

|candidate|GPU|seed|机制侧重|teacher clean/sat/zid|proxy强度/start|soft mix强度|source episode|phase2 radius cap|
|---|---:|---:|---|---|---|---:|---:|---:|
|`EPOC_R7_FLOOR_LOCKED_SHELL`|2|706701|最大化旧类floor保护；虚拟unknown只作轻壳层约束|1.75/0.58/0.320|0.0030/E55|0.00000|0.0078|14|
|`EPOC_R7_BALANCED_LOW_DENSITY`|3|706711|稍强低密度壳层，同时保持teacher/旧类compact|1.55/0.64/0.280|0.0045/E60|0.00002|0.0068|16|

## 本地变更

|文件|目的|sha256|
|---|---|---|
|`E:\type10-7\code\scripts\launch_phase1_epoc_r7_floor_protected_shell_20260706.sh`|新增R7 source-only floor-protected shell训练启动器；经子agent审查后延后proxy启动、降低soft mixup、放宽phase2半径，并修正dry-run日志为source-heldout proxy+virtual unknown shell。|`7F369C0B01FB7A9D46799ACD2F54DCB8DB31074876A08F2EF84B407B490F7D66`|
|`E:\type10-7\code\tests\test_phase1_epoc_r7_floor_protected_shell_launcher.py`|验证dry-run协议边界、候选数量、关键参数和source-heldout proxy声明。|`86DBFD5CC646E31DC3B8741182966B3A2DDC8B824D3F4AE645F76FBBFBDA63F3`|

Snapshot:

`E:\type10-7\code\snapshots\phase1_epoc_r7_floor_protected_shell_20260706`

## 本地验证

|命令|结果|
|---|---|
|`bash -n code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh`|PASS|
|`bash code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh --dry-run --only=EPOC_R7_FLOOR_LOCKED_SHELL`|PASS|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_phase1_epoc_r7_floor_protected_shell_launcher.py code\tests\test_epoc_adv3b02_teacher_distill.py -q`|PASS:4 passed；仅`.pytest_cache`权限warning。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r7_floor_protected_shell_launcher.py`|PASS|

子agent合理性审查后修正：

|修正项|内容|
|---|---|
|proxy启动|从约26epoch推迟到55/60epoch，避免重复R6强proxy失败。|
|soft unknown mixup|主候选降为0，对照仅保留0.00002极低权重。|
|phase2 radius cap|从8/10放宽到14/16，避免过紧prototype半径伤害旧类尾部。|
|teacher distill|提高clean/sat/z_id蒸馏权重，优先保护旧类floor和LEO几何。|
|dry-run声明|从过窄的`virtual_unknown_only`修正为`source_heldout_proxy_unknown=1`和`virtual_unknown_shell=1`，避免误读训练信号来源。|

## 远端计划

|字段|内容|
|---|---|
|remote_root|`/home/szu2070436088/2510044040/CV-SincNet`|
|python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|run_root|`runs/phase1_epoc_r7_floor_protected_shell_20260706`|
|log_root|`logs/phase1_epoc_r7_floor_protected_shell_20260706`|
|launch command|`bash code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh`|
|GPU policy|默认GPU2和GPU3；N607预检后若GPU2/3不再低占用，启动器仍会按`MAX_ACTIVE_PER_GPU=2`等待slot。|
|startup checks|`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；扫描Traceback、RuntimeError、CUDA OOM、NaN、unrecognized arguments。|
|expected outputs|每候选`metrics_epoch.csv`、`metrics_epoch.jsonl`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt`。|

## N607同步与启动证据

|字段|内容|
|---|---|
|preflight|2026-07-06 03:54 CST PASS；direct `N607`可用，project root可见，GPU2/3分别约2185/2179MiB。|
|remote sync|launcher、test、report、`code/SYNC_MANIFEST.txt`已同步到`/home/szu2070436088/2510044040/CV-SincNet`对应路径。|
|remote verify|远端hash匹配，本地/远端`bash -n`、dry-run、测试函数/py_compile检查PASS；dry-run确认ManySig-only、未加载ManyTx、无`--new_wisig_pkl`和真实`target_unknown`。|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; nohup bash code/scripts/launch_phase1_epoc_r7_floor_protected_shell_20260706.sh > logs/phase1_epoc_r7_floor_protected_shell_20260706_driver.out 2>&1 < /dev/null &`|
|driver PID|`3176795`|
|candidate PIDs/logs|`EPOC_R7_FLOOR_LOCKED_SHELL`:PID`3176812`,GPU2,log`logs/phase1_epoc_r7_floor_protected_shell_20260706/EPOC_R7_FLOOR_LOCKED_SHELL.out`;`EPOC_R7_BALANCED_LOW_DENSITY`:PID`3177236`,GPU3,log`logs/phase1_epoc_r7_floor_protected_shell_20260706/EPOC_R7_BALANCED_LOW_DENSITY.out`。|
|startup health|2026-07-06 03:54 CST复查：两个候选均进入`E009/200`；日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`；GPU2/3约2185/2179MiB；未见Traceback、RuntimeError、CUDA OOM、NaN、unrecognized arguments或Killed。|
|claim boundary|当前只证明R7已启动且startup PASS；不得声明Stage2-C成功、部署成功或最终目标完成。训练完成后必须用qknn8与协同推理`M=1..全体目标接收机数量`复评旧类、seen-new和unknown。|

## 运行监控

|time CST|candidate|epoch|best_epoch|best_score|best_val_tx|best_test_tx|latest_val_tx|latest_train_tx|proxy_active|proxy_auc|virtual_accept|source_episode_overflow|proto_export|verdict|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
|2026-07-06 03:59|`EPOC_R7_FLOOR_LOCKED_SHELL`|19/200|10|85.7313|98.6369|90.6623|98.5714|93.3774|0|NA|NA|0.8438|not yet|running; no Stage2-C eval yet|
|2026-07-06 03:59|`EPOC_R7_BALANCED_LOW_DENSITY`|19/200|10|83.4509|98.5536|89.2039|98.5952|91.5745|0|NA|NA|0.7882|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:04|`EPOC_R7_FLOOR_LOCKED_SHELL`|25/200|10|85.7313|98.6369|90.6623|98.6548|91.6466|0|NA|NA|0.8395|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:04|`EPOC_R7_BALANCED_LOW_DENSITY`|25/200|20|84.4508|98.5655|89.7289|98.4881|93.3894|0|NA|NA|0.7673|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:07|`EPOC_R7_FLOOR_LOCKED_SHELL`|29/200|10|85.7313|98.6369|90.6623|98.6369|93.2091|0|NA|NA|0.8392|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:07|`EPOC_R7_BALANCED_LOW_DENSITY`|29/200|20|84.4508|98.5655|89.7289|98.5595|93.6178|0|NA|NA|0.7534|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:10|`EPOC_R7_FLOOR_LOCKED_SHELL`|31/200|10|85.7313|98.6369|90.6623|98.6131|91.9832|0|NA|NA|0.8328|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:10|`EPOC_R7_BALANCED_LOW_DENSITY`|31/200|20|84.4508|98.5655|89.7289|98.4940|94.5192|0|NA|NA|0.7604|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:13|`EPOC_R7_FLOOR_LOCKED_SHELL`|36/200|10|85.7313|98.6369|90.6623|98.5714|92.2596|0|NA|NA|0.8357|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:13|`EPOC_R7_BALANCED_LOW_DENSITY`|37/200|20|84.4508|98.5655|89.7289|98.5298|93.5577|0|NA|NA|0.7675|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:16|`EPOC_R7_FLOOR_LOCKED_SHELL`|39/200|10|85.7313|98.6369|90.6623|98.5893|94.4591|0|NA|NA|0.8334|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:16|`EPOC_R7_BALANCED_LOW_DENSITY`|39/200|20|84.4508|98.5655|89.7289|98.6012|91.0096|0|NA|NA|0.7736|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:20|`EPOC_R7_FLOOR_LOCKED_SHELL`|45/200|10|85.7313|98.6369|90.6623|98.5536|93.6058|0|NA|NA|0.8497|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:20|`EPOC_R7_BALANCED_LOW_DENSITY`|46/200|40|85.0348|98.5893|89.7549|98.5536|94.2668|0|NA|NA|0.7628|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:23|`EPOC_R7_FLOOR_LOCKED_SHELL`|49/200|10|85.7313|98.6369|90.6623|98.6131|93.4135|0|NA|NA|0.8346|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:23|`EPOC_R7_BALANCED_LOW_DENSITY`|49/200|40|85.0348|98.5893|89.7549|98.5714|93.3654|0|NA|NA|0.7598|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:26|`EPOC_R7_FLOOR_LOCKED_SHELL`|52/200|10|85.7313|98.6369|90.6623|98.5655|92.3438|0|NA|NA|0.8337|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:26|`EPOC_R7_BALANCED_LOW_DENSITY`|53/200|40|85.0348|98.5893|89.7549|98.5119|93.1010|0|NA|NA|0.7795|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:30|`EPOC_R7_FLOOR_LOCKED_SHELL`|58/200|10|85.7313|98.6369|90.6623|98.5833|93.7019|1|0.4747|0.8137|0.8388|not yet|running; early proxy weak/negative|
|2026-07-06 04:30|`EPOC_R7_BALANCED_LOW_DENSITY`|59/200|40|85.0348|98.5893|89.7549|98.5595|91.5625|0|NA|NA|0.7798|not yet|running; no Stage2-C eval yet|
|2026-07-06 04:35|`EPOC_R7_FLOOR_LOCKED_SHELL`|63/200|10|85.7313|98.6369|90.6623|98.6131|93.0889|1|0.4746|0.8163|0.8419|not yet|running; proxy remains weak/negative|
|2026-07-06 04:35|`EPOC_R7_BALANCED_LOW_DENSITY`|64/200|60|NA|98.6190|90.0064|98.5060|93.0409|1|0.4395|0.8107|0.7690|not yet|running; proxy also weak/negative|
|2026-07-06 04:38|`EPOC_R7_FLOOR_LOCKED_SHELL`|68/200|10|85.7313|98.6369|90.6623|98.5655|93.2692|1|0.4832|0.8173|0.8352|not yet|running; proxy remains below random|
|2026-07-06 04:38|`EPOC_R7_BALANCED_LOW_DENSITY`|69/200|60|NA|98.6190|90.0064|98.6607|92.8726|1|0.4378|0.8085|0.7846|not yet|running; proxy remains clearly negative|
|2026-07-06 04:41|`EPOC_R7_FLOOR_LOCKED_SHELL`|70/200|10|85.7313|98.6369|90.6623|98.5833|93.3534|1|0.4800|0.8127|0.8362|not yet|running; E070 test_tx=89.6936, proxy remains weak|
|2026-07-06 04:41|`EPOC_R7_BALANCED_LOW_DENSITY`|70/200|60|NA|98.6190|90.0064|98.6071|94.2428|1|0.4394|0.8122|0.7717|not yet|running; E070 test_tx=89.8279, proxy remains negative|
|2026-07-06 04:44|`EPOC_R7_FLOOR_LOCKED_SHELL`|73/200|10|85.7313|98.6369|90.6623|98.6488|92.1274|1|0.4726|0.8166|0.8471|not yet|running; proxy remains weak/negative|
|2026-07-06 04:44|`EPOC_R7_BALANCED_LOW_DENSITY`|74/200|60|NA|98.6190|90.0064|98.6012|93.9904|1|0.4405|0.8127|0.7590|not yet|running; proxy remains negative|
|2026-07-06 04:46|`EPOC_R7_FLOOR_LOCKED_SHELL`|77/200|10|85.7313|98.6369|90.6623|98.6250|94.3269|1|0.4814|0.8139|0.8284|not yet|running; proxy still below random|
|2026-07-06 04:46|`EPOC_R7_BALANCED_LOW_DENSITY`|78/200|60|NA|98.6190|90.0064|98.5179|92.9567|1|0.4361|0.8122|0.7737|not yet|running; proxy still negative|
|2026-07-06 04:49|`EPOC_R7_FLOOR_LOCKED_SHELL`|79/200|10|85.7313|98.6369|90.6623|98.6012|93.1010|1|0.4793|0.8115|0.8380|not yet|running; proxy still below random|
|2026-07-06 04:49|`EPOC_R7_BALANCED_LOW_DENSITY`|80/200|60|NA|98.6190|90.0064|98.4940|92.4639|1|0.4368|0.8088|0.7731|not yet|running; E080 test_tx=88.7936, proxy still negative|
|2026-07-06 04:55|`EPOC_R7_FLOOR_LOCKED_SHELL`|89/200|10|85.7313|98.6369|90.6623|98.6071|93.4135|1|0.4800|0.8113|0.8352|not yet|running; proxy still below random|
|2026-07-06 04:55|`EPOC_R7_BALANCED_LOW_DENSITY`|89/200|60|85.5133|98.6190|90.0064|98.6190|93.5457|1|0.4371|0.8127|0.7629|not yet|running; proxy still negative|

03:59 CST只读监控结论：N607预检PASS，GPU2/3分别约2443/2437MiB；两个候选均有`best_joint_safe_ssdg.pth`，但尚未导出`phase2_zid_prototypes.pt`。错误扫描未见Traceback、RuntimeError、CUDA OOM、out-of-memory、NaN、unrecognized arguments或Killed。由于proxy未知计划从E55/E60才启动，当前`proxy_active=0`属于预期，不可据此判断unknown拒识效果。当前状态仍是训练中，不能启动Stage2-C qknn8协同复评。

04:04 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选均继续运行到E025/200，仍未导出`phase2_zid_prototypes.pt`。`BALANCED_LOW_DENSITY`在E020刷新best_score到84.4508；`FLOOR_LOCKED_SHELL`best仍为E010。错误扫描仍为空。当前仍不满足Stage2-C复评前置条件。

04:07 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选均继续运行到E029/200，仍早于proxy启动点E55/E60，`proxy_active=0`仍属预期。两个候选均未导出`phase2_zid_prototypes.pt`，错误扫描为空，Stage2-C qknn8协同复评继续延后。

04:10 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选均继续运行到E031/200，仍早于proxy启动点E55/E60。`FLOOR_LOCKED_SHELL`best仍停留E010，`BALANCED_LOW_DENSITY`best仍停留E020；两个候选均未导出`phase2_zid_prototypes.pt`，错误扫描为空，Stage2-C qknn8协同复评继续延后。

04:13 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`到E036/200，`BALANCED_LOW_DENSITY`到E037/200，仍早于proxy启动点E55/E60，`proxy_active=0`仍属预期。两个候选均未导出`phase2_zid_prototypes.pt`，错误扫描为空，Stage2-C qknn8协同复评继续延后。

04:16 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选均到E039/200，仍早于proxy启动点E55/E60，`proxy_active=0`仍属预期。`FLOOR_LOCKED_SHELL`best仍为E010，`BALANCED_LOW_DENSITY`best仍为E020；两个候选均未导出`phase2_zid_prototypes.pt`。日志尾部只命中字符串`nan`，来自普通日志文本扫描，不对应训练崩溃；未见Traceback、RuntimeError或CUDA OOM。Stage2-C qknn8协同复评继续延后。

04:20 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`到E045/200，`BALANCED_LOW_DENSITY`到E046/200，仍早于proxy启动点E55/E60。`BALANCED_LOW_DENSITY`在E040刷新best_score到85.0348；`FLOOR_LOCKED_SHELL`best仍为E010。两个候选均有`best_joint_safe_ssdg.pth`，均未导出`phase2_zid_prototypes.pt`。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；精确文本扫描仅命中`nan_token`，不构成崩溃证据。Stage2-C qknn8协同复评继续延后。

04:23 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选均到E049/200，仍早于proxy启动点E55/E60，`proxy_active=0`仍属预期。两个候选均有`best_joint_safe_ssdg.pth`和`latest_safe_ssdg.pth`，均未导出`phase2_zid_prototypes.pt`；日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed，精确文本扫描仅命中`nan_token`。Stage2-C qknn8协同复评继续延后，下一次重点检查E55/E60之后的proxy指标和prototype导出。

04:26 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`到E052/200，`BALANCED_LOW_DENSITY`到E053/200。二者仍早于各自proxy启动点E55/E60，`proxy_active=0`仍属预期；最近5个epoch的proxy AUC、virtual accept和proxy accept字段仍为空。两个候选均未导出`phase2_zid_prototypes.pt`，日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed，精确文本扫描仅命中`nan_token`。Stage2-C qknn8协同复评继续延后。

04:30 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`到E058/200且已过proxy启动点E55，proxy字段开始产生，但E055-E058的proxy AUC约0.4728-0.4837，E058为0.4747，virtual accept约0.8137，proxy accept约0.0115，属于早期弱/负向proxy趋势；`BALANCED_LOW_DENSITY`到E059/200，仍早于自身proxy启动点E60，`proxy_active=0`仍属预期。两个候选均未导出`phase2_zid_prototypes.pt`，日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed，精确文本扫描仅命中`nan_token`。Stage2-C qknn8协同复评继续延后；下一次重点检查`BALANCED_LOW_DENSITY` E060之后proxy是否也呈现弱AUC/高virtual accept。

04:35 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`到E063/200，E056-E063的proxy AUC约0.4728-0.4837，E063为0.4746，virtual accept约0.8163，说明弱/负向proxy趋势没有改善；`BALANCED_LOW_DENSITY`到E064/200，E060-E064的proxy AUC约0.4339-0.4402，E064为0.4395，virtual accept约0.8107，且soft unknown mixup virtual accept约0.9981，说明它在proxy启动后同样没有形成有效未知分离。两个候选均未导出`phase2_zid_prototypes.pt`，日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前结论仍是训练中负向趋势，而不是最终Stage2-C失败；必须等待训练完成和prototype导出后再做qknn8协同复评。

04:38 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`主进程PID`3176812`仍运行，到E068/200，最近E064-E068的proxy AUC约0.4743-0.4832，E068为0.4832，virtual accept约0.8173，仍低于随机分离水平；`BALANCED_LOW_DENSITY`主进程PID`3177236`仍运行，到E069/200，最近E065-E069的proxy AUC约0.4335-0.4417，E069为0.4378，virtual accept约0.8085，soft unknown mixup virtual accept约0.9968。两个候选均未导出`phase2_zid_prototypes.pt`，日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。Stage2-C qknn8协同复评继续延后；当前只形成“R7 proxy分离持续负向”的过程证据。

04:41 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；两个候选主进程仍运行并到E070/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL` E070的test_tx为89.6936，低于当前best E010的90.6623，proxy AUC为0.4800，virtual accept为0.8127；`BALANCED_LOW_DENSITY` E070的test_tx为89.8279，低于当前best E060的90.0064，proxy AUC为0.4394，virtual accept为0.8122，soft unknown mixup virtual accept为0.9994。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前过程证据继续支持“proxy未知分离没有形成”，但Stage2-C qknn8协同复评仍必须等prototype导出后执行。

04:44 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`主进程PID`3176812`仍运行，到E073/200，仍未导出`phase2_zid_prototypes.pt`，最近E069-E073的proxy AUC约0.4726-0.4841，E073为0.4726，virtual accept约0.8166；`BALANCED_LOW_DENSITY`主进程PID`3177236`仍运行，到E074/200，仍未导出`phase2_zid_prototypes.pt`，最近E070-E074的proxy AUC约0.4343-0.4405，E074为0.4405，virtual accept约0.8127，soft unknown mixup virtual accept约0.9987。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前仍只能记录R7训练中负向proxy趋势，Stage2-C qknn8协同复评继续等待prototype导出。

04:46 CST只读监控结论：N607预检PASS，GPU2/3分别约2463/2459MiB；`FLOOR_LOCKED_SHELL`主进程PID`3176812`仍运行，到E077/200，仍未导出`phase2_zid_prototypes.pt`，最近E073-E077的proxy AUC约0.4726-0.4835，E077为0.4814，virtual accept约0.8139；`BALANCED_LOW_DENSITY`主进程PID`3177236`仍运行，到E078/200，仍未导出`phase2_zid_prototypes.pt`，最近E074-E078的proxy AUC约0.4361-0.4423，E078为0.4361，virtual accept约0.8122，soft unknown mixup virtual accept约0.9981。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。本次SSH命令结束后发现本地`ssh.exe`残留PID`34968`和到N607:22的ESTABLISHED连接，已按规则关闭并复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。当前仍不能进入Stage2-C，继续等待prototype导出。

04:49 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；`FLOOR_LOCKED_SHELL`主进程PID`3176812`仍运行，到E079/200，仍未导出`phase2_zid_prototypes.pt`，最近E075-E079的proxy AUC约0.4783-0.4835，E079为0.4793，virtual accept约0.8115；`BALANCED_LOW_DENSITY`主进程PID`3177236`仍运行，到E080/200，仍未导出`phase2_zid_prototypes.pt`，E080的test_tx为88.7936，低于当前best E060的90.0064，最近E076-E080的proxy AUC约0.4361-0.4423，E080为0.4368，virtual accept约0.8088，soft unknown mixup virtual accept约0.9981。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。Stage2-C qknn8协同复评继续等待prototype导出；R7当前过程证据仍为proxy未知分离失败趋势。

04:55 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行，`FLOOR_LOCKED_SHELL`到E089/200，`BALANCED_LOW_DENSITY`到E089/200，均仍未导出`phase2_zid_prototypes.pt`。按CSV真实字段`train_proxy_unknown_auc_proxy`复核后，`FLOOR_LOCKED_SHELL`最近E085-E089的proxy AUC约0.4741-0.4834，E089为0.4800，virtual accept约0.8113；`BALANCED_LOW_DENSITY`最近E085-E089的proxy AUC约0.4330-0.4417，E089为0.4371，virtual accept约0.8127，soft unknown mixup virtual accept约0.9962。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；`nan`命中来自`sat_cos=nan`、`aux=nan`和非test epoch的`overall_tx=nan% (0/0)`，不构成训练崩溃证据。Stage2-C qknn8协同复评继续等待prototype导出；R7过程证据继续显示proxy未知分离没有形成。

## 成功/失败判据

|阶段|判据|
|---|---|
|startup PASS|两个候选均启动，日志进入epoch，未见配置/argparse/OOM错误。|
|training useful|同row`best_score`不低于EPOC B，且`sat_floor`/receiver floor不明显退化。|
|进入Stage2-C条件|训练完成并导出prototype后，必须用Stage2-C qknn8+协同推理复评`M=1..5`，真实unknown只作eval。|
|失败判据|若旧类floor仍未明显改善，则R7也进入负证据；下一步需改变模型/损失结构，而不是继续微调壳层权重。|
