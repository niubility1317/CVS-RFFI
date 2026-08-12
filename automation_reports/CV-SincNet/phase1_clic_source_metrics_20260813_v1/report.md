# Phase1 CLIC源域指标补全v1报告

## Task1预注册：source-V单观测LEO缓存

- 实验ID：`phase1_clic_source_metrics_20260813_v1`。
- 当前状态：`LOCAL_VERIFYING / NO_PERFORMANCE_RESULT`。
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
