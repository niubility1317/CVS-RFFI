# Phase2单物理样本单LEO接收观测追踪表

日期：2026-07-17

状态：协议先行修订

## 1. 变更原因

旧D1/D3路线把同一个clean/raw物理IQ样本分别叠加`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，再把三份观测用于Phase2适配或评估。该构造不符合单颗卫星接收机的实际可达信息：一次接收只能得到一个既定LEO星地信道下的IQ观测，不能同时取得同一发射波形在三种场景下的平行观测。

本次先修改`项目.md`，再据此审计和修复cache builder、sealed package、validator、runner、实验矩阵与报告。旧结果只保留为历史诊断，不得继续用于候选选择、超参数锁定、正式排名或性能声明。

## 2. 需求到实现追踪

|ID|协议需求|`项目.md`落点|实现/验证状态|
|---|---|---|---|
|SO-01|一个clean/raw物理IQ样本在进入Phase2前必须且只能叠加一次，并恰好叠加一个`leo_*_weak`场景，形成唯一接收IQ观测|第7.1.1节|本次落盘|
|SO-02|同一matched receiver×seed×K×new规模下，三个场景的support∪query全角色物理样本并集两两不交，同场景support/query也不交|第7.1.1节、第10.3.1节|cache、lineage v2、authority v2、offline package和两个bundle loader均已实现并通过组合测试|
|SO-03|Phase2密封包必须逐样本记录不可重命名的pre-overlay稳定根ID、唯一scenario、恰好一个satellite seed值和overlay provenance|第7.1.1节|稳定根已改为dataset SHA+原始WiSig坐标；offline package按scenario独立生成opaque token并绑定逐样本overlay provenance|
|SO-04|禁止从同一clean/raw样本派生多场景、多信道或多子样本用于Phase2训练、适配、注册、校准、选择或评估|第7.1.1节|本次落盘；历史D1/D3需降级|
|SO-05|允许的多view只能由已接收的固定LEO_weak IQ在Phase2内执行接收侧信道均衡、增强、变换或表征提取；不得重新访问clean/raw、叠加另一LEO状态或恢复另一物理观测|第7.1.1节、第8节、第10.3.1节|D4a原语和scenario-atomic diag已实现；真实new5/10/20开发行已运行|
|SO-06|同一接收IQ派生的计算view不得计作独立support或增加K；只有support view可参与拟合，query view只能用于当前样本推理且不得更新任何状态|第7.1.1节、第10.3.1节|D4a输出逐view parent IQ SHA/operator/view seed；K只计物理support，query fit/update为0|
|SO-07|历史D1/D3因确认使用同一物理样本的三种LEO观测，统一标记`PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION`|第7.1.1节、第12节|本次落盘；报告待更新|
|SO-08|极轻型正式上限调整为adapter参数不超过80,000、适配不超过30epoch、持久状态不超过256KB、无dense query图|第10.3.1节|统一runtime contract、method lock和D4a resource receipt已接入；D4a为864参数、0epoch、状态低于95KB|
|SO-09|三场景仍需完整覆盖，但它们是三个独立接收样本单元，不是同一物理样本的三份view|第8节、第10.3.1节|TX×day分层互斥分配和post-build validator已完成；真实cache三scenario各1040行、交集0|
|SO-10|offline authority可维护每类最多20个候选support，但Phase2密封package与loader对每类只能暴露当前row的exact K；K1/K5/K10不得可达更大候选池|第7节K-shot报告口径第230–231行|verified：formal matrix v3明确分离`offline_authority_support_pool_max_k=20`与`reachable_support_pool_max_k=k_shot`|
|SO-11|每个exact-K package必须在打开IQ前核验成员allowlist和逐类可达support计数；`reachable_support_pool_max_k>K`必须阻断|第7节K-shot报告口径第230–231行|verified：row强制allowlist与pre-open逐类计数，validator要求可达上限和逐类计数均严格等于row K|
|SO-12|嵌套K必须验证K1=对应K10有序support前1、K5=前5；K20使用独立只暴露20个support/类的package|第7节K-shot报告口径第231行|verified：formal row锁定`k1_k5_are_k10_ordered_prefixes_k20_is_separate_exact_package`并强制前缀验证|
|SO-13|任一额外support可达、allowlist缺失、pre-open逐类计数不等于K或嵌套前缀不符，统一标记`PROTOCOL_INVALID_KSHOT_REACHABILITY`并阻断正式row|第7节K-shot报告口径第230–231行|verified：状态常量、schema精确键和重签名负向测试均已落地|

## 3. 合法与非法view边界

合法：

- K-shot由K个互不重复的独立物理样本构成；例如K5是5份独立接收IQ，不是一个样本的5个副本。
- 从唯一已接收LEO_weak IQ执行信道均衡、时频增强、接收侧变换，或计算固定时域、频域、时频和RF统计表征。
- 对唯一已接收IQ执行预登记、接收侧可计算的变换，并将多个分支用于同一个物理样本的联合表征、一致性约束或轻量适配。
- 所有分支共享同一个`physical_sample_id`、scenario、satellite seed和support/query角色；K只计一次。
- 每个派生view记录`parent_received_iq_sha256`、`operator_id`和`view_seed`；operator不得调用LEO channel simulator或创建新的overlay provenance。
- 只有support派生view可以参与适配或状态更新；query派生view只服务当前query的逐样本推理。

非法：

- 从同一clean/raw IQ分别叠加三种LEO场景。
- 从同一clean/raw IQ生成多份带不同LEO状态或不同信道随机性的子样本。
- 把同一接收IQ的计算分支当作多个独立support或独立query，从而放大K或样本数。
- 在Phase2中重新打开clean/raw IQ、叠加另一LEO状态、逆推出clean参考或生成另一物理接收观测。

## 4. 旧结果边界

D1和D3的已保存score、loss trace、逐类结果与资源数据继续保留，作为“旧三场景同源多观测构造为何会产生表观性能”的历史诊断证据。但它们不再满足当前正式Phase2单观测协议，不能用于：

- 选择下一candidate或超参数；
- 证明K10/K5/K1性能；
- 形成125确认矩阵；
- 与identity-only单qKNN做正式Pareto排名；
- 声明Stage2-B/C部署性能或floor达标。

## 5. 当前验证与后续

已完成：

1. 真实`rx20-1/seed713101`cache三场景各1040行，跨scenario物理根交集0。
2. lineage receipt为`BYTE_GROUNDED_SELF_CONSISTENCY_PASS`，single-observation与cross-scenario audit均PASS。
3. offline package、authority v2、SOMP-H/通用bundle loader均改为scenario独立选择并两两互斥。
4. D4a真实K10 new5/10/20注册前后开发行完整结束，864参数、0epoch、无query fit。

当前性能失败边界：

- D4a注册前old_acc为76.39%。
- new5/10/20注册后old_acc为62.22%/57.78%/55.56%。
- seen-new为68.67%/59.33%/50.50%，旧类遗忘随新类数从14.17pp扩大至20.83pp。

因此下一轮不是扩大125，而是先完成D4b old-head lock、support-only floor guard和稳健多prototype修复；达到开发门槛后再锁定candidate并扩展K1/K5和正式125矩阵。

## 6. exact-K生产闭包验证

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|EK-01|`项目.md`第230行|离线候选池max20不得等同于Phase2可达support池|`code/cvsrffi/somph_formal_matrix.py`|verified|formal matrix单测与268组回归通过|保留旧`support_pool_max_k`仅作offline authority兼容别名，并增加不可误解的语义字段|
|EK-02|`项目.md`第230行|K1/K5/K10/K20 row逐类严格只可达自身K|`code/cvsrffi/somph_formal_matrix.py`、`tests/test_somph_formal_matrix.py`|verified|重签名后把K1可达池改为20会以`PROTOCOL_INVALID_KSHOT_REACHABILITY`拒绝|validator同时要求`reachable_support_pool_max_k==reachable_support_count_per_registered_class==k_shot`|
|EK-03|`项目.md`第231行|IQ打开前验证成员allowlist、逐类计数和嵌套前缀|`code/cvsrffi/somph_formal_matrix.py`、`tests/test_somph_formal_matrix.py`|verified|allowlist、pre-open count、prefix字段逐项负向测试通过|formal matrix只声明不可降级的生产要求，实际package/bundle闭包由同组回归验证|
|EK-04|`项目.md`第230–231行|缺失或冲突时使用统一协议无效状态|`code/cvsrffi/somph_formal_matrix.py`、`tests/test_somph_formal_matrix.py`|verified|精确schema与重签名guard-drift测试通过|统一状态为`PROTOCOL_INVALID_KSHOT_REACHABILITY`|

验证命令：

```powershell
conda activate ssr-gpu
python -m pytest tests/test_somph_formal_matrix.py -q
python -m pytest code/tests/test_build_cvs_leo_weak_iq_cache.py code/tests/test_leo_weak_cache.py code/tests/test_phase2_runtime_contract.py tests/test_build_cvs_stage2_predictor_bundle.py tests/test_sign_cvs_somph_authority_lock.py tests/test_somph_authority_lock_builder.py tests/test_somph_cache_build_matrix.py tests/test_somph_formal_matrix.py tests/test_somph_leo_weak_lineage_seal.py tests/test_somph_lineage_authority.py tests/test_somph_offline_target_package.py tests/test_somph_predictor_bundle.py tests/test_stage2_predictor_bundle.py tests/test_somph_predictor_runtime.py tests/test_somph_stage2c.py -q
git diff --check -- code/cvsrffi/somph_formal_matrix.py tests/test_somph_formal_matrix.py analysis/phase2_single_observation_traceability_20260717.md
```

结果：单文件`19 passed`；原268组回归加本次新增6个exact-K测试后共`274 passed`；`git diff --check`通过，仅有工作区既有LF/CRLF提示。
