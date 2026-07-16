# D17-SPRTDR support-only runner追踪表

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|D17R-01|`项目.md`§7.1/§7.1.1|只打开sealed LEO_weak enrollment receipt/feature；pre-open先验哈希与allowlist审计；不得触达clean/source|`code/scripts/run_d17_support_only_sprtdr.py`、runner tests|verified|复用D14`_load_enrollment`、`_payload_rows`、`_build_feature_artifact`；静态NO_QUERY测试通过|外部expected seal SHA为必填CLI|
|D17R-02|单物理样本单观测协议|exact-K按互异`physical_sample_id`计数；同样本不得多overlay；K1为canonical Z0；K5/K10为独立真实物理样本|runner、tests|verified|K1/K5/K10 exact-K正例与重复physical ID反例通过；K1仅返回rank0/force-zero|开发选择入口只运行strict K10；K1/K5不得用于超参选择|
|D17R-03|跨场景协议|三场景support的物理ID和`parent_received_iq_sha256`均两两零重叠|runner、tests|verified|两类交集均审计；parent SHA复用反例fail closed|按after enrollment union检查，覆盖old+new|
|D17R-04|D17候选锁|候选仅Z0、margin 0.02、margin 0.04，统一锁定|runner、tests|verified|候选值、确定性lock SHA测试通过|未修改D17核心|
|D17R-05|support-only选择门|outer L2O的每个scene×fold逐类门；old/new同等优先；old floor与new floor分别严格收益；禁止均值抵消；显式检查old→new决策遗忘|runner、tests|verified|单fold单class下降、new-floor无严格收益、旧score锁定但旧决策遗忘三类反例均阻断|旧score列锁不能替代旧类决策门|
|D17R-06|真Z0回退|所有positive arm未同时通过三场景全部fold门时选择canonical true Z0|runner、tests|verified|全部positive失败回退测试通过；无enabled edge也不能通过正臂门|不得把无边positive state冒充收益|
|D17R-07|禁止query/正式声明|CLI和运行时不接受query/truth/scorer/authority；固定development diagnostic support-only|runner、tests|verified|CLI源码表面检查及formal authority false检查通过|不产生formal authority|
|D17R-08|完整证据|保存append-only JSONL、support audit、receipt、报告及state/MAC/时延/内存Pareto审计|runner、tests|verified|输出路径与字段静态检查、`py_compile`及runner+module测试通过|闭式0参数、0epoch、无dense query图|
|D17R-09|serializer红队P0|每场景selected before/after均保存不可覆盖`state.npz`+`metadata.json`+`COMMIT`；禁止pickle；加载后执行成员bytes/SHA、`SprtdrState`语义、state SHA及固定probe score逐位验证|runner、tests|verified|canonical NPZ round-trip、错误外部COMMIT SHA、重复输出路径与禁pickle测试通过|6份真实state写入各自独立目录；总实际文件字节进入support audit|

验证：`conda run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_sprtdr.py code/scripts/run_d17_support_only_sprtdr.py`；D17 core+runner为26 passed；再联合`tests/test_stage2_fcar.py`与`tests/test_stage2_sparse_pairwise_fisher_guard.py`共50 passed。D17 core SHA=`49BAFA32F1789949533CC152A0649ECDD478CCF704470B295CE98CCB6691BE6A`；runner SHA=`59A1933ECC21D19016A8F6B000D808EAC11B10F1B54D57CD60CEAC104DA53190`。

真实sealed strict-K10 v3已完成：`d17_sprtdr_strict_k10_v3_final`统一选择canonical true Z0；三场景old-score逐位锁均为true，六份state总79,343B、单份最大15,232B，全部round-trip通过；`query/truth/scorer=false`。因此协议、资源与state证据GO，性能路线NO-GO。
