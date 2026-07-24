# D103-R2需求—实现—测试—artifact追踪

状态：`DESIGN_FROZEN_REV3 / IMPLEMENTING_LOCAL_ONLY / N607_NO_GO / TARGET25_NO_GO`

|ID|需求|设计依据|实现文件|验证或artifact|当前状态|
|---|---|---|---|---|---|
|R2-01|全局精确0.07/0.63/0.30正式分离归档|重入卡§3|`code/cvsrffi/rxid_metabias4_source_archive.py`、`code/scripts/export_d103_r1_source_splits.py`|待实现42个receiver×TX各14条L、day 2–4、任一leave-day后K10可达；U=5292、V=2520|revision-pending|
|R2-02|跨receiver K1/K5/K10元任务|重入卡§2|待修订`rxid_metabias4_phase1_trainer.py`|真实计数min10/max13；修订后真实一步smoke|pending|
|R2-03|R1其余Phase1机制保持|重入卡§2|现有trainer|TX-null/MMD/selfsup/VICReg测试|implemented|
|R2-04|M0/D102/D103 matched scorer|重入卡§4|计划held runner|同support/query；49个`L_s` fold-specific D102诊断bundle；原reject receipt绑定|pending|
|R2-05|4 leave-day实际shift方向门|重入卡§4|计划held runner/gate|196个day fit completion；`cos(B_day a_day,B_outer a_outer)`；近零fail closed|pending|
|R2-06|K1数值、INT8和双TX probe门|R1卡§5–7；R2卡§4|现有Stage2/bundle/falsifier|focused tests已实现，整体验证待完成|implemented|
|R2-07|246fit/98,400step非覆盖runner|R2卡§5|计划正式runner/pipeline|失败fingerprint、逐fit状态、资源receipt|pending|
|R2-08|Target25严格25行单seed|用户当前口径|计划gate后matrix|5receiver×1seed×5slice；held gate前禁止|blocked|

最高风险：跨receiver元任务必须先通过独立设计复审；不得在审查前把R1常量静默改写为R2。
