# Phase1 CLIC源域指标补全v1报告

## Task1预注册：source-V单观测LEO缓存

- 实验ID：`phase1_clic_source_metrics_20260813_v1`。
- 当前状态：`SMOKE_STOPPED_TECHNICAL_GATE / FORMAL_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`。
- 本任务只创建`source_validation_known_leo_weak`的16,800行received-IQ缓存构建器；它不同于既有3,920行source-L尾部校准缓存，绝不重建、修改或读取后者的结果。
- 该工件属于`POST_TARGET_COMPLETION_AUDIT_NON_SELECTION`：目标端确认已经封存。本任务不得以任何方式使用目标truth、指标、候选排序、阈值、重训、重试、复活或晋级决策。

## 冻结输入与输出合同

| 项目 | 冻结合同 |
|---|---|
| 训练身份 | `phase1_clic12_20260812_v5`；同一fold的`F* C_CLIC12`与`F* G_CLIC12`最终checkpoint及terminal receipt |
| clean证据 | `phase1_clic_postfreeze_20260812_v4`；两臂`source_clean_proxy.npz`的V索引、物理metadata/order及manifest必须逐项一致 |
| V角色 | `source_validation_known_leo_weak`；仅内部local4 held-V，精确16,800行；不加载外部held TX |
| 物理规则 | 同一物理ID只生成一个received-IQ；在每个`(tx_id,rx_id)`内按不透明ID稳定排序、round-robin分配`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak` |
| C/G共享 | 同fold C/G只消费同一个`F*_SHARED/source_validation_known_leo_weak.npz`；缓存receipt记录两臂输入SHA和相同字节绑定 |
| 信道与seed | 复用source-LEO冻结参数；`checkpoint.seed+991+scene_index×1,000,003` |
| 输出 | `runs/phase1_clic_source_metrics_20260813_v1/F*_SHARED/{source_validation_known_leo_weak.npz,source_validation_known_leo_weak.receipt.json}`；拒绝覆盖和非canonical路径 |

## 访问与技术边界

- `fit_rows=0`、`threshold_fit_rows=0`、`source_v_forward_rows=0`、`proxy_forward_rows=0`、`target_access=false`、`query_access=false`、`selection_access=false`、`retry_access=false`。
- 构建前、V物化后和receipt写入前均复算checkpoint、terminal、clean-v4和WiSig输入SHA；任何TOCTOU漂移或非有限received-IQ都不能保留部分输出。
- 输出NPZ只含`received_iq`、`tx_ids`、`rx_ids`、`day_ids`、`physical_sample_id`与`sat_scenarios`。无需且不得产生性能指标。
- 本Task1不进行N607预检、同步或启动；后续Task3经独立审查和冻结release后才有资格交给唯一runner。

## 本地实现与验证

- 受控文件：`code/build_phase1_clic_source_v_leo_iq.py`、`code/tests/test_build_phase1_clic_source_v_leo_iq.py`及本报告。
- 测试先行范围：精确V重建、L/V/proxy物理互斥、稳定场景分配、C/G共享、输入/输出不变性、非有限信道拒绝、TOCTOU拒绝与安全Torch/NumPy桥接。
- RED：生产模块不存在时，`ssr-gpu`聚焦pytest按预期报`ModuleNotFoundError: build_phase1_clic_source_v_leo_iq`；未以测试替身绕过该缺失边界。
- GREEN：`python -m pytest -q code/tests/test_build_phase1_clic_source_v_leo_iq.py`通过`11/11`；包括不可覆盖和非canonical输出负例。
- 静态核验：`py_compile`、构建器CLI`--help`和`git diff --check`均通过。
- 技术缓存闭合不等于source指标通过，更不等于Phase1晋级；尚未发生N607同步或正式实验启动。

## Task1 P1竞态修复：不可覆盖发布与发布后核验

- 复审触发条件：原`temporary.replace(path)`在临时文件写完与最终路径落地之间可覆盖并发创建的不可变NPZ或receipt；原`temporary.exists()`与`open("xb")`／`open("x")`之间的竞争还可能在异常清理时删除外部`.tmp`。
- RED证据：新增动态竞态测试后，旧实现分别暴露最终NPZ／receipt哨兵被覆盖、外部NPZ有效替换未被拒绝、以及NPZ／receipt外部临时文件被删除。
- 修复：临时文件仅在本方成功独占创建后记录`device/inode`身份；同目录写入并`fsync`后以`os.link`独占创建最终名称，绝不使用替换式发布。任何冲突均失败关闭。清理仅在路径仍与本方记录身份一致时执行，因此不会删除并发所有者的最终文件、替换文件或临时文件。
- 发布后核验：NPZ在重开验证、输入哈希检查、receipt封存前后均核对发布身份与SHA；receipt发布后和返回前同样核对身份与SHA。有效但外部替换的NPZ也会被拒绝，且外部文件保持原样。
- GREEN：`ssr-gpu`下`python -m pytest -q code/tests/test_build_phase1_clic_source_v_leo_iq.py`通过`16/16`；其中5项为最终路径／临时路径并发安全回归。该修复仍不读取目标端、不发起N607操作，也不改变缓存的数据、场景或指标语义。

## Task1 P1补充：同inode原地篡改与写失败清理

- 新RED：仅核对`device/inode`不足。攻击者可在最终名通过`os.link`后以`r+b`原地写入另一份仍可重开、字段自洽的NPZ，或把receipt JSON中的`target_access`改为`true`，两者inode均不变。旧实现因在最终路径发布后才首次取SHA而错误采纳篡改字节。
- 修复：`_ImmutablePublication`现封存临时文件写完并`fsync`后的SHA；`os.link`前核临时身份和该SHA，最终NPZ及receipt的每次发布后核验均与此预封SHA比较。发布后不得以路径当前内容重新建立基线。
- 清理边界：最终已发布工件只在身份和预封SHA都一致时删除；若检测到同inode原地篡改则保留可疑外部内容。仅本方已经成功独占创建、但尚未形成可封SHA的临时文件，在写入异常时按身份清理，避免遗留`.tmp`永久阻塞下一次安全运行。
- GREEN：同inode有效NPZ篡改、同inodereceipt篡改和写中途异常临时清理均通过；完整Task1聚焦测试现为`19/19`。这仍是本地工件完整性修复，未运行N607或产生任何性能结果。

## Task2本地实现：source-V单次forward与同checkpoint指标

- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。本节实现本地file-only前向与评分器；尚未运行N607、同步文件、读取目标工件或产生任何性能数值。
- 新增工件：`code/export_phase1_clic_source_v_leo_features.py`和`code/evaluate_phase1_clic_source_metrics.py`。前者对每个`F*{C,G}_CLIC12`只消费同fold的`source_validation_known_leo_weak`缓存；后者只消费两臂冻结的clean-v4、V-forward、PAIR-v3和terminal/checkpoint收据。
- forward重新打开training-v5最终checkpoint、terminal receipt、同臂clean-v4 metadata/manifest、Task1 V缓存/receipt与PAIR-v3。clean-v4仅通过Task1已有metadata-only重开器验证L/V/proxy物理互斥和held-V身份，不读取或评分source-L/proxy特征行。
- source-V缓存与clean-v4须同时满足同一V索引SHA、V完整metadata顺序SHA，以及逐行TX/RX/day/sig一致；forward负载再逐行绑定缓存的TX/RX/day/physical/scene，`raw_labels`必须等于冻结local4类别顺序。任何TOCTOU、非有限数、重复physical ID、类顺序、轴或场景错位均失败关闭。
- PAIR-v3只读复用既有source-L geometry/tail policy和fixed400 proxy诊断。每个场景policy必须与PAIR的`single_leo_common_binding`的received-IQ/physical-order SHA一致；所有V/proxy指标路径均保持`fit_rows=0`与`threshold_fit_rows=0`。
- V前向使用单个固定received-IQ DataLoader，`shuffle=false`、`satellite_tta_policy=none`、`received_existing`，每个physical ID恰好一次。进入Torch和返回NumPy都使用安全buffer/list桥接，不调用`torch.from_numpy`或`Tensor.numpy`。
- feature NPZ和binding、fold pair metrics receipt以及six-fold aggregate receipt都使用Task1的预封SHA、同目录无覆盖发布和发布后identity/SHA核验；所有新receipt均写入`POST_TARGET_COMPLETION_AUDIT_NON_SELECTION`，不能用于训练、阈值、选择、重试、复活或晋级。

## 指标与门禁合同

| 切片 | 正确性 | 封存字段 |
|---|---|---|
| clean-V | 唯一local4`argmax(tx_logits)==truth` | overall、macro、class/RX/day原始correct/denominator、四个minimum floor |
| 每个LEO场景V | `decision=registered`且冻结local4预测等于truth；`unknown`/`defer`均计错 | 与clean-V相同的原始cells/floors及known unknown/defer错误计数 |
| proxy | 仅PAIR-v3的`AUROC_unknown`、`u_gap` | C/G双侧零fit/threshold，要求`delta_AUROC>0`和`delta_u_gap>0` |

- 每fold的clean及三个formal scene均逐项检查overall/min-class/min-RX/min-day的`G-C>=-2pp`。
- 每fold的三scene等权overall以及完整`6×3=18`scene等权overall均要求`>=-2pp`。原始axis分子/分母必须各自回加到overall，不能由不一致或零分母的cell补偿。
- gate不通过只返回`passed=false`，不会触发重试或停止；该source证据也不能补偿已失败的target-real-unknown门。

## Task2测试与本地验证

- RED已在生产API缺失时记录为`ModuleNotFoundError: evaluate_phase1_clic_source_metrics`；随后每个新增绑定、原始axis一致性、非整数标签、cache/clean身份和PAIR policy/common-binding负例均先观察到失败，再最小实现GREEN。
- 当前聚焦：`python -m pytest -q code/tests/test_phase1_clic_source_metrics.py`通过`17/17`。覆盖cache/receipt哈希漂移、target访问标志、V-only角色、单physical单forward、旧Torch/NumPy桥接禁止、非有限和非整数输入、known unknown/defer计错、零分母/角色/scene复用、clean/cache/feature元数据绑定、每fold/全18门禁、严格proxy增益、CLI`--help`，以及pair-score编排中的每臂clean-v4 SHA传递和不可覆盖输出。
- 受影响回归：`python -m pytest -q code/tests/test_phase1_clic_postfreeze.py code/tests/test_phase1_clic_common_receipt_export.py`全绿；仅保留既有PyTorch`autocast`弃用警告。
- 静态验证：新模块`py_compile`、两份CLI`--help`和`git diff --check`均通过。当前没有N607运行、目标访问或性能结果。

## Task3可追溯预注册：12臂source指标release

- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。本阶段只编排已经封存的training-v5、clean-v4、PAIR-v3和source-V缓存输入；不会读取target、query truth、target package或prediction输入。
- run ID固定为`phase1_clic_source_metrics_20260813_v1`。输出根和日志根均为新建、不可覆盖路径；retry为`NO`，任何性能数值不参与调度、停止、重试、选择、复活或晋级。

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| T3-01 | Task3 | 严格构建6份同fold C/G共享source-V缓存 | 新launcher、launcher测试 | verified | dry-run计数与路径断言 | 缓存先于任何consumer完成，单fold只产生一个`F*_SHARED`工件对 |
| T3-02 | Task3 | 生成12份C/G source-V forward，所有forward传对应clean-v4 NPZ | 新launcher、launcher测试 | verified | dry-run逐臂参数断言 | 每physical row的单次forward由既有exporter执行 |
| T3-03 | Task3 | 生成12臂fold C/G metrics证据和1份六foldaggregate | 新launcher、launcher测试 | verified | dry-run逐fold参数与aggregate断言 | 6份pair receipt各封C/G两臂；scorer只读PAIR-v3 proxy并在CPU执行 |
| T3-04 | Task3 | 固定12臂映射、6共享cache、forward每GPU最多2项、scorer CPU | 新launcher、launcher测试 | verified | dry-run GPU/CPU环境断言 | 不改变训练、公式、阈值或冻结方法 |
| T3-05 | Task3 | fresh roots、拒绝覆盖、无target/query/truth/prediction输入 | 新launcher、launcher测试 | verified | dry-run禁止输入断言和非dry-run根保护 | source-only及`POST_TARGET_COMPLETION_AUDIT_NON_SELECTION`只由既有工件封存 |
| T3-06 | Task3 | 预注册命令、路径、PID台账、预期工件、健康停止和retry=NO | 本报告 | verified | 报告复核 | 不访问N607，不报告性能结果 |

### Task3本地验证与N607交接合同

- 实现提交：`08276ab4b87c5558c370572fa33fe272b5a5d169`；此前真实物理ID修复提交：`416c2498182f9b1fcc0db97fe74fe4685481bf32`。
- 文件SHA256：launcher=`3F143E888B04DB373B4CE5C39906817818501D747E28DF103A264848CF917EEE`；launcher测试=`743B4C347E623F395314FA458BA6AC69F0F545A0FDF71D1EB9A06C86475FE7BF`。
- 本地验证：`ssr-gpu`下launcher行为测试`3/3`通过；`bash -n`通过；dry-run精确`25`行=`6 cache+12 forward+6 pair score+1 aggregate`，禁止的target/query/truth/prediction/package/retry命中`0`；`git diff --check`通过。
- 固定输入：training=`runs/phase1_clic12_20260812_v5`、clean=`runs/phase1_clic_postfreeze_20260812_v4`、PAIR=`runs/phase1_clic_source_pair_20260812_v3`、WiSig SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 正式输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260813_v1`；日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260813_v1`；outer：`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v1_outer.out`。
- 正式入口：`nohup bash <immutable-release>/code/scripts/launch_phase1_clic_source_metrics12_v1_20260813.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v1_outer.out 2>&1 &`；环境为release内`CODE_ROOT`、`CVS-RFFI` Python、正式C/G forward使用GPU0..5且每卡最多2个并发consumer，score/aggregate强制CPU。
- 启动前必须做direct N607 preflight、release/正式run/log/outer均ABSENT、输入和SHA闭合。强制先在独立smoke root执行F1共享cache+C/G consumer结构smoke；smoke失败则formal invocation保持0。smoke通过后才允许唯一formal invocation=1，retry=`NO`。
- 健康停止：仅协议/SHA/覆盖/访问违规、launcher级确定性失败，或至少两个不同row在产出receipt前出现同一归一化异常指纹时停止精确run-owned进程树；绝不因accuracy、proxy AUROC或任何性能值停止、重试或选择。预期正式工件为6个共享cache+receipt、12个feature NPZ+binding、6个pair receipt、1个aggregate、PID表和分阶段日志。

## Task2 P1修复可追溯记录：source-V physical ID语义

- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。本修复只纠正Task2对Task1 source-V缓存`physical_sample_id`的重开语义；不修改scorer、Task1 builder、训练、缓存、target或N607路径。

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| P1-01 | Task1物理ID合同 | 正例必须通过Task1`_physical_sample_id(dataset_sha256,tx,rx,day,eq,sig)`构造，不将clean raw`sig`当作缓存physical ID | exporter、聚焦测试 | verified | RED后GREEN | 生产代码只调用Task1 helper，不复制哈希公式 |
| P1-02 | Task2 pre-forward绑定 | 从clean-v4 manifest封存的dataset SHA及逐行V`tx/rx/day/eq/sig`重算expected ID，并与缓存逐行精确比较 | exporter、聚焦测试 | verified | 六字段漂移负例 | `sig`仍参与独立ID绑定 |
| P1-03 | 安全边界 | dataset/tx/rx/day/eq/sig任一漂移必须在model forward和任何输出发布前拒绝 | exporter、聚焦测试 | verified | 无输出断言 | 保留scene/feature/TOCTOU/zero-access合同 |

### P1根因、RED/GREEN与边界证据

- 根因：`dff5686a`中的Task2 exporter将Task1 metadata-hash`physical_sample_id`直接与clean-v4 raw`validation_sig_ids`逐行比较。二者不是同一命名空间，因此真实Task1正例被错误拒绝。
- RED：先将正例改为真实Task1`_physical_sample_id(dataset_sha256,tx,rx,day,eq,sig)`构造。`ssr-gpu`下运行`python -m pytest -q code/tests/test_phase1_clic_source_metrics.py -k source_v_forward_reopens_clean_v4_identity_before_forward`按预期失败于`source-V cache/clean-v4 physical_ids row binding drifted`。
- 修复：exporter从Task1 metadata-only clean-v4 binding中的manifest`wisig_pkl_sha256`及逐行`validation_tx_ids/validation_rx_ids/validation_day_ids/validation_eq_ids/validation_sig_ids`调用Task1 helper重算expected physical ID；缓存ID必须逐行完全相等。TX/RX/day原轴绑定继续保留，预期ID重复也失败关闭。
- GREEN：真实正例加dataset/TX/RX/day/eq/sig六个漂移预检共`7/7`通过；每个漂移均在PAIR加载、model forward和immutable输出发布之前拒绝，`source_v_features.npz`和binding均不存在。dataset-SHA变异检查会使漂移路径越过绑定层并到达PAIR桩，恢复读取manifest后的实现重新通过。
- 回归：`code/tests/test_phase1_clic_source_metrics.py`共`23/23`通过；`test_phase1_clic_postfreeze.py`与`test_phase1_clic_common_receipt_export.py`共`171/171`通过。`py_compile`、exporter/scorer`--help`和`git diff --check`均通过。
- 边界：没有N607、target、query truth、性能计算、输出artifact或参数选择；本记录仅证明本地P1身份绑定闭合，不构成source指标、训练质量或任何晋级结论。

## F1独立结构smoke封存：技术失败

- 状态：`SMOKE_STOPPED_TECHNICAL_GATE / FORMAL_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`。
- `SMOKE_INVOCATION=1`，唯一worker PID=`754555`；F1 cache builder子PID=`754626`，exit=`1`；`retry=NO`，`FORMAL=0`。
- 精确trace fingerprint：`CLICSplitExportError: CLIC terminal envelope selected checkpoint path drifted`。
- 根因：独立smoke tree中的hardlink镜像路径与terminal envelope记录的absolute selected checkpoint路径不一致，builder在checkpoint/terminal重开阶段拒绝继续。这是smoke路径绑定失败，不是性能或协议结果。
- F1C/F1G exporter均未启动；生成物为`0 cache/receipt/feature/binding`，不执行score，不读取accuracy、AUROC或任何gate。

### Smoke路径与输入镜像

- smoke root：`/home/szu2070436088/2510044040/CV-SincNet/.smoke_phase1_clic_source_metrics_20260813_v1_F1`。
- worker：`<smoke root>/smoke_worker.sh`；outer log：`<smoke root>/logs/smoke_worker.outer.out`；builder log：`<smoke root>/logs/phase1_clic_source_metrics_20260813_v1/F1_CACHE_BUILDER.out`（2382 bytes）；manifest：`<smoke root>/logs/phase1_clic_source_metrics_20260813_v1/smoke_manifest.tsv`。
- 6个不可变输入均优先hardlink成功，镜像前后SHA一致；PAIR F1与ManySig均直接读取正式v3/正式原件，未复制或修改：

| 输入 | bytes | SHA256 |
|---|---:|---|
| F1C checkpoint | 7681196 | `eebc810879f32c1db83f05a5b47794c804468b3e718e85b42cac034935d1aa01` |
| F1C terminal | 1523377 | `ee923c9d767891e60f826a5d332b797926ac6a0ac96b56b2cbf17f52f508a697` |
| F1C clean | 32126888 | `ff7166a2dd0b70455711ce7997ffd385534aa9a4cda8fe280e810dca7702f86c` |
| F1G checkpoint | 7681196 | `a0b5cb61d4ef922d1446cd93f870c29f4f4c51050e36c775c02c462d51415be5` |
| F1G terminal | 1523389 | `bd75d3c08bda7f2f4a456423885d21e3c4f94251bb22deb3966eec4e8336dcd1` |
| F1G clean | 32126884 | `c68aa2bfd8ba4d3c9adc1e76c1b1b2ee76416d79305c0a1a42b11d48e511009e` |

### Release、远端静态与安全边界证据

- immutable release：`/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v1_175b1540`；archive physical=`268113920` bytes，SHA256=`27e96ef020d3e3e014df40817f8cf15fd4d42825ac15acba838378d4c08debad`；SCP恰一次，upload tar与该SHA一致。
- 远端静态检查：`py_compile=PASS`、三个`--help=PASS`、`bash -n=PASS`、dry-run=`25`（cache=`6`、forward=`12`、pair=`6`、aggregate=`1`），forbidden target/query/truth/prediction/package/`--retry`均为`0`。
- 正式`run`、`log`、`outer`在smoke失败后仍为`ABSENT`；GPU0为`0%/1MiB`，相关进程为`0`，本地SSH/SCP进程与TCP22连接均为`0`。smoke root及全部partial log保留，未删除、未覆盖。

后续只能创建新的run ID `phase1_clic_source_metrics_20260813_v2`重新完成独立路径合同；本v1不得retry、不得改标为formal、不得产生性能结论。
