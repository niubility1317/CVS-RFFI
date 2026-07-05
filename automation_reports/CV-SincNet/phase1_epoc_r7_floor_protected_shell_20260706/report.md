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
|status|n607_running_monitor_e180_e179|

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
|2026-07-06 04:59|`EPOC_R7_FLOOR_LOCKED_SHELL`|91/200|10|85.7313|98.6369|90.6623|98.5714|94.5553|1|0.4846|0.8192|0.8328|not yet|running; E090 test_tx=89.7632, proxy still below random|
|2026-07-06 04:59|`EPOC_R7_BALANCED_LOW_DENSITY`|92/200|90|85.8023|98.6012|90.0270|98.5774|93.3053|1|0.4395|0.8071|0.7623|not yet|running; best refreshed at E090 but proxy still negative|
|2026-07-06 05:08|`EPOC_R7_FLOOR_LOCKED_SHELL`|103/200|10|85.7313|98.6369|90.6623|98.6310|93.3293|1|0.4826|0.8175|0.8485|not yet|running; E100 test_tx=89.6647, proxy still below random|
|2026-07-06 05:08|`EPOC_R7_BALANCED_LOW_DENSITY`|103/200|90|85.8023|98.6012|90.0270|98.5833|94.6995|1|0.4407|0.8050|0.7611|not yet|running; E100 test_tx=89.9412, proxy still negative|
|2026-07-06 05:15|`EPOC_R7_FLOOR_LOCKED_SHELL`|111/200|10|85.7313|98.6369|90.6623|98.6429|93.6058|1|0.4783|0.8255|0.8403|not yet|running; E110 test_tx=89.7382, proxy still below random|
|2026-07-06 05:15|`EPOC_R7_BALANCED_LOW_DENSITY`|111/200|90|85.8023|98.6012|90.0270|98.6429|92.8606|1|0.4347|0.8141|0.7646|not yet|running; E110 test_tx=89.0892, proxy still negative|
|2026-07-06 05:19|`EPOC_R7_FLOOR_LOCKED_SHELL`|117/200|10|85.7313|98.6369|90.6623|98.6071|92.3077|1|0.4841|0.8171|0.8567|not yet|running; proxy still below random|
|2026-07-06 05:19|`EPOC_R7_BALANCED_LOW_DENSITY`|116/200|90|85.8023|98.6012|90.0270|98.5774|93.8221|1|0.4375|0.8088|0.7563|not yet|running; proxy still negative|
|2026-07-06 05:24|`EPOC_R7_FLOOR_LOCKED_SHELL`|122/200|10|85.7313|98.6369|90.6623|98.5952|94.4712|1|0.4770|0.8207|0.8379|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:24|`EPOC_R7_BALANCED_LOW_DENSITY`|122/200|90|85.8023|98.6012|90.0270|98.6250|95.0841|1|0.4442|0.8038|0.7600|not yet|running; prototype absent, proxy still negative|
|2026-07-06 05:28|`EPOC_R7_FLOOR_LOCKED_SHELL`|128/200|10|85.7313|98.6369|90.6623|98.6845|94.1346|1|0.4830|0.8175|0.8407|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:28|`EPOC_R7_BALANCED_LOW_DENSITY`|128/200|90|85.8023|98.6012|90.0270|98.5655|91.9591|1|0.4317|0.8111|0.7826|not yet|running; prototype absent, proxy still negative|
|2026-07-06 05:32|`EPOC_R7_FLOOR_LOCKED_SHELL`|130/200|10|85.7313|98.6369|90.6623|98.6250|93.3413|1|0.4722|0.8243|0.8295|not yet|running; E130 test_tx=89.5157, prototype absent|
|2026-07-06 05:32|`EPOC_R7_BALANCED_LOW_DENSITY`|130/200|90|85.8023|98.6012|90.0270|98.6369|92.0553|1|0.4388|0.8106|0.7789|not yet|running; E130 test_tx=89.9343, prototype absent|
|2026-07-06 05:35|`EPOC_R7_FLOOR_LOCKED_SHELL`|135/200|10|85.7313|98.6369|90.6623|98.5774|90.6370|1|0.4749|0.8168|0.8400|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:35|`EPOC_R7_BALANCED_LOW_DENSITY`|135/200|90|85.8023|98.6012|90.0270|98.5774|93.8101|1|0.4407|0.8175|0.7696|not yet|running; prototype absent, proxy still negative|
|2026-07-06 05:39|`EPOC_R7_FLOOR_LOCKED_SHELL`|139/200|10|85.7313|98.6369|90.6623|98.6012|92.0673|1|0.4723|0.8171|0.8438|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:39|`EPOC_R7_BALANCED_LOW_DENSITY`|139/200|90|85.8023|98.6012|90.0270|98.6190|94.6995|1|0.4397|0.8109|0.7542|not yet|running; prototype absent, proxy still negative|
|2026-07-06 05:54|`EPOC_R7_FLOOR_LOCKED_SHELL`|159/200|10|85.7313|98.6369|90.6623|98.6488|93.7861|1|0.4788|0.8132|0.8419|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:54|`EPOC_R7_BALANCED_LOW_DENSITY`|158/200|90|85.8023|98.6012|90.0270|98.6369|94.2308|1|0.4400|0.8101|0.7646|not yet|running; prototype absent, proxy still negative|
|2026-07-06 05:58|`EPOC_R7_FLOOR_LOCKED_SHELL`|162/200|10|85.7313|98.6369|90.6623|98.4762|92.8726|1|0.4762|0.8197|0.8424|not yet|running; prototype absent, proxy still below random|
|2026-07-06 05:58|`EPOC_R7_BALANCED_LOW_DENSITY`|161/200|90|85.8023|98.6012|90.0270|98.6190|92.8726|1|0.4411|0.8080|0.7753|not yet|running; prototype absent, proxy still negative|
|2026-07-06 06:01|`EPOC_R7_FLOOR_LOCKED_SHELL`|167/200|10|85.7313|98.6369|90.6623|98.6131|94.6514|1|0.4847|0.8139|0.8307|not yet|running; prototype absent, proxy still below random|
|2026-07-06 06:01|`EPOC_R7_BALANCED_LOW_DENSITY`|167/200|90|85.8023|98.6012|90.0270|98.6250|92.1755|1|0.4386|0.8095|0.7773|not yet|running; prototype absent, proxy still negative|
|2026-07-06 06:06|`EPOC_R7_FLOOR_LOCKED_SHELL`|171/200|10|85.7313|98.6369|90.6623|98.6488|93.1010|1|0.4796|0.8120|0.8423|not yet|running; prototype absent, proxy still below random|
|2026-07-06 06:06|`EPOC_R7_BALANCED_LOW_DENSITY`|171/200|90|85.8023|98.6012|90.0270|98.6310|94.7236|1|0.4386|0.8133|0.7667|not yet|running; prototype absent, proxy still negative|
|2026-07-06 06:13|`EPOC_R7_FLOOR_LOCKED_SHELL`|175/200|10|85.7313|98.6369|90.6623|98.6131|93.6899|1|0.4808|0.8200|0.8350|not yet|running; prototype absent, proxy still below random|
|2026-07-06 06:13|`EPOC_R7_BALANCED_LOW_DENSITY`|175/200|90|85.8023|98.6012|90.0270|98.6071|93.8462|1|0.4408|0.8067|0.7610|not yet|running; prototype absent, proxy still negative|
|2026-07-06 06:21|`EPOC_R7_FLOOR_LOCKED_SHELL`|180/200|10|85.7313|98.6369|90.6623|98.6548|92.7644|1|0.4835|0.8135|0.8394|not yet|running; E180 test_tx=89.2299 below best, prototype absent|
|2026-07-06 06:21|`EPOC_R7_BALANCED_LOW_DENSITY`|179/200|90|85.8023|98.6012|90.0270|98.6071|93.0409|1|0.4373|0.8056|0.7677|not yet|running; prototype absent, proxy still negative|

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

04:59 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行，`FLOOR_LOCKED_SHELL`到E091/200，`BALANCED_LOW_DENSITY`到E092/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E087-E091的proxy AUC约0.4748-0.4846，E091为0.4846，virtual accept约0.8192，E090 test_tx为89.7632，仍低于best E010的90.6623；`BALANCED_LOW_DENSITY`在E090刷新best到85.8023，best_test_tx为90.0270，较E060的90.0064只小幅提高，但E092 proxy AUC仍为0.4395，最近E088-E092约0.4361-0.4417，virtual accept约0.8071，soft unknown mixup virtual accept约0.9955。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；`BALANCED_LOW_DENSITY`日志中出现一次`[GRAD] total=nan backbone=nan aux=nan domain=3.907`，但进程和后续metrics仍继续推进，应作为数值稳定性观察项而非崩溃证据。Stage2-C qknn8协同复评继续等待prototype导出；R7未知proxy分离仍未形成。

05:08 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行并均到E103/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E099-E103的proxy AUC约0.4722-0.4856，E103为0.4826，virtual accept约0.8175，E100 test_tx为89.6647，继续低于best E010的90.6623；`BALANCED_LOW_DENSITY`最近E099-E103的proxy AUC约0.4365-0.4407，E103为0.4407，virtual accept约0.8050，soft unknown mixup virtual accept约0.9974，E100 test_tx为89.9412，低于best E090的90.0270。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；本次尾部错误扫描未再出现`total=nan`，仅有`sat_cos=nan`、`aux=nan`和非test epoch的`overall_tx=nan% (0/0)`。Stage2-C qknn8协同复评继续等待prototype导出；R7过程证据仍显示proxy未知分离没有形成。

05:15 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行并均到E111/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E107-E111的proxy AUC约0.4739-0.4847，E111为0.4783，virtual accept约0.8255，E110 test_tx为89.7382，仍低于best E010的90.6623；`BALANCED_LOW_DENSITY`最近E107-E111的proxy AUC约0.4324-0.4415，E111为0.4347，virtual accept约0.8141，soft unknown mixup virtual accept约0.9987，E110 test_tx为89.0892，明显低于best E090的90.0270。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；尾部扫描仍只见`sat_cos=nan`、`aux=nan`和非test epoch的`overall_tx=nan% (0/0)`文本。Stage2-C qknn8协同复评继续等待prototype导出；R7过程证据继续显示proxy未知分离没有形成。

05:19 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行，`FLOOR_LOCKED_SHELL`到E117/200，`BALANCED_LOW_DENSITY`到E116/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E113-E117的proxy AUC约0.4772-0.4841，E117为0.4841，virtual accept约0.8171；`BALANCED_LOW_DENSITY`最近E112-E116的proxy AUC约0.4348-0.4418，E116为0.4375，virtual accept约0.8088，soft unknown mixup virtual accept约0.9962。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed；尾部扫描仍只见`sat_cos=nan`、`aux=nan`和非test epoch的`overall_tx=nan% (0/0)`文本。Stage2-C qknn8协同复评继续等待prototype导出；R7过程证据仍未显示有效未知分离。

05:24 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行并均到E122/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E118-E122的proxy AUC约0.4758-0.4781，E122为0.4770，virtual accept约0.8207，proxy accept约0.0113；`BALANCED_LOW_DENSITY`最近E118-E122的proxy AUC约0.4336-0.4442，E122为0.4442，virtual accept约0.8038，soft unknown mixup virtual accept约0.9968。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前仍不能启动Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。

05:28 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2459MiB；两个候选主进程仍运行并均到E128/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E124-E128的proxy AUC约0.4726-0.4830，E128为0.4830，virtual accept约0.8175，proxy accept约0.0058；`BALANCED_LOW_DENSITY`最近E124-E128的proxy AUC约0.4317-0.4431，E128为0.4317，virtual accept约0.8111，soft unknown mixup virtual accept约0.9962。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。Stage2-C qknn8协同复评继续等待prototype导出；R7过程证据仍为未知proxy分离不足。

05:32 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2461MiB；两个候选主进程仍运行并均到E130/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E126-E130的proxy AUC约0.4722-0.4830，E130为0.4722，virtual accept约0.8243，E130 test_tx为89.5157，低于best E010的90.6623；`BALANCED_LOW_DENSITY`最近E126-E130的proxy AUC约0.4317-0.4434，E130为0.4388，virtual accept约0.8106，soft unknown mixup virtual accept约0.9955，E130 test_tx为89.9343，低于best E090的90.0270。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前仍不能启动Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。

05:35 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2461MiB；GPU0/1/4/5/6/7为空闲显存低占用。两个R7候选主进程仍运行并均到E135/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E131-E135的proxy AUC约0.4749-0.4835，E135为0.4749，virtual accept约0.8168；`BALANCED_LOW_DENSITY`最近E131-E135的proxy AUC约0.4365-0.4425，E135为0.4407，virtual accept约0.8175，soft unknown mixup virtual accept约0.9987。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前仍不能启动Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。本次本地SSH清理时发现既有残留`ssh.exe` PID 22292，命令为早前`phase2_qknn_hardpair_n20_aligned_20260706`的nohup launch通道；已关闭本地残留进程并复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

05:39 CST只读监控结论：N607预检PASS，GPU2/3分别约2465/2461MiB，GPU0/1/4/5/6/7仍为空闲低显存。两个R7候选主进程仍运行并均到E139/200，均仍未导出`phase2_zid_prototypes.pt`。`FLOOR_LOCKED_SHELL`最近E135-E139的proxy AUC约0.4723-0.4817，E139为0.4723，virtual accept约0.8171；`BALANCED_LOW_DENSITY`最近E135-E139的proxy AUC约0.4338-0.4435，E139为0.4397，virtual accept约0.8109，soft unknown mixup virtual accept约0.9981。日志未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。

05:54 CST只读监控结论：N607直连短命令检查显示GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；`find`未发现任何`phase2_zid_prototypes.pt`。为避免宽进程列表污染判断，本次只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0554`解析。`FLOOR_LOCKED_SHELL`到E159/200，best仍为E010，E159 proxy AUC为0.4788、virtual accept为0.8132、source episode overflow为0.8419；`BALANCED_LOW_DENSITY`到E158/200，best仍为E090，E158 proxy AUC为0.4400、virtual accept为0.8101、soft unknown mixup virtual accept为0.9981、source episode overflow为0.7646。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。一次远端监控命令因PowerShell提前展开远端变量而超时，本地残留SSH PID`30400`已关闭并复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`；后续短命令与SCP后也复查清理干净。

05:58 CST只读监控结论：N607预检PASS，GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；R7两个主PID仍运行，`find`仍未发现`phase2_zid_prototypes.pt`。只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0558`解析。`FLOOR_LOCKED_SHELL`到E162/200，best仍为E010，E162 proxy AUC为0.4762、virtual accept为0.8197、source episode overflow为0.8424；`BALANCED_LOW_DENSITY`到E161/200，best仍为E090，E161 proxy AUC为0.4411、virtual accept为0.8080、soft unknown mixup virtual accept为0.9968、source episode overflow为0.7753。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。SSH/SCP后均复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

06:01 CST只读监控结论：N607预检PASS，GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；R7两个主PID仍运行，`find`仍未发现`phase2_zid_prototypes.pt`。只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0601`解析。`FLOOR_LOCKED_SHELL`到E167/200，best仍为E010，E167 proxy AUC为0.4847、virtual accept为0.8139、source episode overflow为0.8307；`BALANCED_LOW_DENSITY`到E167/200，best仍为E090，E167 proxy AUC为0.4386、virtual accept为0.8095、soft unknown mixup virtual accept为0.9987、source episode overflow为0.7773。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。本轮同时复核Stage2-C入口：基础全量复评应使用`code/scripts/phase2_collaborative_open_set_qknn_eval.py --collab_counts all --qknn_k 8`覆盖M=1..target receiver count；`phase2_tcsr_ci_eval.py`和`phase2_socapr_qknn8_pareto_eval.py`仅作为后续决策层诊断封装，不替代基础qknn8全量复评。SSH/SCP后均复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

06:06 CST只读监控结论：N607预检PASS，GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；R7两个主PID仍运行，`find`仍未发现`phase2_zid_prototypes.pt`，错误扫描未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0606`解析。`FLOOR_LOCKED_SHELL`到E171/200，best仍为E010，E171 proxy AUC为0.4796、virtual accept为0.8120、source episode overflow为0.8423；`BALANCED_LOW_DENSITY`到E171/200，best仍为E090，E171 proxy AUC为0.4386、virtual accept为0.8133、soft unknown mixup virtual accept为0.9994、source episode overflow为0.7667。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。SSH/SCP后均复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

06:13 CST只读监控结论：N607预检PASS，GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；R7两个主PID仍运行到E175/200，`find`仍未发现`phase2_zid_prototypes.pt`，错误扫描未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0613`解析。`FLOOR_LOCKED_SHELL`best仍为E010，E175 proxy AUC为0.4808、virtual accept为0.8200、source episode overflow为0.8350；`BALANCED_LOW_DENSITY`best仍为E090，E175 proxy AUC为0.4408、virtual accept为0.8067、soft unknown mixup virtual accept为0.9974、source episode overflow为0.7610。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。SSH/SCP后均复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

06:21 CST只读监控结论：N607预检PASS，GPU2/3分别约2669/2665MiB，GPU0/1/4/5/6/7仍约10MiB；R7两个主PID仍运行，`FLOOR_LOCKED_SHELL`到E180/200，`BALANCED_LOW_DENSITY`到E179/200，`find`仍未发现`phase2_zid_prototypes.pt`或Stage2 feature npz，错误扫描未见Traceback、RuntimeError、CUDA OOM、out-of-memory、unrecognized arguments或Killed。只读拉取两个`metrics_epoch.csv`到本地临时目录`E:\type10-7\local_artifacts\r7_metrics_20260706_0621`解析。`FLOOR_LOCKED_SHELL`best仍为E010，E180 test_tx为89.2299，低于best_test_tx 90.6623，proxy AUC为0.4835、virtual accept为0.8135、source episode overflow为0.8394；`BALANCED_LOW_DENSITY`best仍为E090，E179 proxy AUC为0.4373、virtual accept为0.8056、soft unknown mixup virtual accept为0.9981、source episode overflow为0.7677。当前仍不能启动R7 Stage2-C qknn8协同推理全量复评；R7过程证据继续显示未知proxy分离不足。SSH/SCP后均复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。

## 成功/失败判据

|阶段|判据|
|---|---|
|startup PASS|两个候选均启动，日志进入epoch，未见配置/argparse/OOM错误。|
|training useful|同row`best_score`不低于EPOC B，且`sat_floor`/receiver floor不明显退化。|
|进入Stage2-C条件|训练完成并导出prototype后，必须用Stage2-C qknn8+协同推理复评`M=1..5`，真实unknown只作eval。|
|失败判据|若旧类floor仍未明显改善，则R7也进入负证据；下一步需改变模型/损失结构，而不是继续微调壳层权重。|
|Stage2-C基础入口|`code/scripts/phase2_collaborative_open_set_qknn_eval.py --collab_counts all --qknn_k 8`；后续可追加TCSR/SOCAPR诊断，但不能替代基础qknn8全量复评。|

## 05:44 Sync Note

05:44 CST执行本地到N607同步：`report.md`同步到`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase1_epoc_r7_floor_protected_shell_20260706/report.md`，并用本地`Get-FileHash`和远端`sha256sum`核对一致。同步后发现既有本地`ssh.exe`残留PID`15320`，命令为早前`phase2_qknn_hardpair_n20_aligned_ref_20260706`的HP08REF nohup launch通道；已只关闭本地SSH客户端并复查为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。
