# D103-R1需求—实现—测试—artifact追踪

状态：`LOCAL_TECHNICAL_INFEASIBLE / NO_PERFORMANCE_RESULT / N607_NOT_RUN / TARGET25_NO_GO`

|ID|需求|设计依据|实现文件|验证或artifact|当前状态|
|---|---|---|---|---|---|
|R1-01|tap+dual hash和同row绑定|重入卡§2|`code/cvsrffi/rxid_metabias4_feasibility_probe.py`|7个focused tests；R1 probe JSON|verified|
|R1-02|L_s/U_s/source-val权限隔离|重入卡§3|`code/cvsrffi/rxid_metabias4_phase1_trainer.py`、`code/scripts/run_d103_r1_phase1_fit.py`|split-negative、exact manifest和access-ledger测试待整体验证|implemented|
|R1-03|rank-5 TX零空间+MMD|重入卡§4、设计草案§3|`code/cvsrffi/rxid_metabias4_phase1_trainer.py`|null residual、MMD和L_s-only测试待整体验证|implemented|
|R1-04|跨day/cross-TX receiver保持|重入卡§3–4|`code/cvsrffi/rxid_metabias4_phase1_trainer.py`|R1 probe42/42可构造；leave-day和U_s无TX测试待整体验证|implemented|
|R1-05|K1/K5/K10同式MetaBias4|重入卡§4–5|`code/cvsrffi/stage2_rxid_metabias4.py`|机械门、inactive闭包和D102等价测试待整体验证|implemented|
|R1-06|INT8-only学习数组|重入卡§7|`code/cvsrffi/rxid_metabias4_bundle.py`|ABI、RNE、饱和、wire roundtrip/bitflip/truncate测试待整体验证|implemented|
|R1-07|query只读、全类独立qKNN|项目.md；重入卡§7|`code/cvsrffi/stage2_rxid_metabias4.py`|repeat/hash、query-fit=0和全类竞争负测待整体验证|implemented|
|R1-08|双TX probe一次性reject|重入卡§6|`code/cvsrffi/rxid_metabias4_held_falsifier.py`、`code/scripts/run_d103_r1_held_gate.py`|physical split、9容量×5评分面和mean/max测试待整体验证|implemented|
|R1-09|7 receiver+42双留出+day审计|重入卡§8|`code/cvsrffi/rxid_metabias4_held_falsifier.py`|246-fit计划和63性能row覆盖测试待整体验证|implemented|
|R1-10|matched M0/D102/D103完整指标|设计草案§6|`code/cvsrffi/rxid_metabias4_held_falsifier.py`|gate聚合已实现；真实fit/predict/score编排仍缺|pending|
|R1-11|资源和失败封口|重入卡§8|feasibility probe、trainer、bundle、Stage2 wrapper|GPUh/disk估算已验证；formal runner资源与失败artifact仍缺|pending|
|R1-12|非覆盖正式run|本表|计划`code/scripts/run_d103_r1_phase1_held.py`、`code/scripts/run_d103_r1_phase1held_pipeline.sh`|预留run ID`d103_r1_rxid_phase1held_20260724_r1`|pending|
|R1-13|L_s/U_s独立source archive及manifest|重入卡§3|`code/cvsrffi/rxid_metabias4_source_archive.py`、`code/scripts/export_d103_r1_source_splits.py`|正式source_train全池尚未构建；结构与负测已通过|implemented|
|R1-14|deployment bundle严格反序列化|重入卡§7|`code/cvsrffi/rxid_metabias4_bundle.py`|canonical wire、成员顺序、content root和预测等价测试待整体验证|implemented|

R1真实特征smoke在第一个optimizer step前失败：正式7%切分后receiver×TX只有10–13条，而R1实现要求同receiver内`K10+query16=26`条。该run未接N607、未产生性能结果；后续仅允许按`docs/D103_R2_RXID_CROSSRECEIVER_REENTRY_CARD.md`重入，R1不得原地改参重跑。
