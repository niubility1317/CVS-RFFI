# D103-R1需求—实现—测试—artifact追踪

状态：`DESIGN_FROZEN / IMPLEMENTING_LOCAL_ONLY`

|ID|需求|设计依据|实现文件|验证或artifact|当前状态|
|---|---|---|---|---|---|
|R1-01|tap+dual hash和同row绑定|重入卡§2|`code/cvsrffi/rxid_metabias4_feasibility_probe.py`|7个focused tests；R1 probe JSON|feasibility-verified|
|R1-02|L_s/U_s/source-val权限隔离|重入卡§3|计划`code/cvsrffi/rxid_metabias4_phase1_trainer.py`|计划split-negative tests和access ledger|planned|
|R1-03|rank-5 TX零空间+MMD|重入卡§4、设计草案§3|计划同上|计划null residual、双probe和outer receipt|planned|
|R1-04|跨day/cross-TX receiver保持|重入卡§3–4|计划同上|R1 probe42/42可构造；计划leave-day tests|partially-verified|
|R1-05|K1/K5/K10同式MetaBias4|重入卡§4–5|计划`code/cvsrffi/stage2_rxid_metabias4.py`|计划机械门、OOF tail和inactive closure tests|planned|
|R1-06|INT8-only学习数组|重入卡§7|计划`code/cvsrffi/rxid_metabias4_bundle.py`|计划ABI golden vector、round/saturation、无sidecar tests|planned|
|R1-07|query只读、全类独立qKNN|项目.md；重入卡§7|复用现有typed qKNN；计划Stage2 wrapper|计划repeat/hash、query-fit=0、全类竞争negative tests|planned|
|R1-08|双TX probe一次性reject|重入卡§6|计划`code/cvsrffi/rxid_metabias4_held_falsifier.py`|计划physical split、容量、mean/max聚合golden tests|planned|
|R1-09|7 receiver+42双留出+day审计|重入卡§8|计划同上|计划49-fold coverage receipt|planned|
|R1-10|matched M0/D102/D103完整指标|设计草案§6|计划同上|计划old/new proxy、H、floor、forgetting、net correct table|planned|
|R1-11|资源和失败封口|重入卡§8|计划runner/pipeline|R1 feasibility review；计划GPUh/disk/state/MAC receipts|partially-verified|
|R1-12|非覆盖正式run|本表|计划`code/scripts/run_d103_r1_phase1_held.py`、`code/scripts/run_d103_r1_phase1held_pipeline.sh`|预留run ID`d103_r1_rxid_phase1held_20260724_r1`|planned|

正式实现只能在独立设计终审允许`DESIGN_FROZEN→IMPLEMENTING`后开始。实现完成后必须把所有`planned`改为`verified`，通过真实checkpoint无query smoke、独立`P0=0/P1=0`、Git commit和runner handoff；否则不得接N607。
