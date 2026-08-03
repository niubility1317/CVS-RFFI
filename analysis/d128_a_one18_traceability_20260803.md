# D128-A-ONE18实现追踪

来源：`58ee10f5`的`docs/STAGE2_RD_GOAL_20260731.md`第5.1节。

|ID|来源要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|
|D128-01|只保留A=`DA-A-FSRG-time_fuse`，不得要求B/C或merged bundle|`stage2_d128_a_one18.py`、`run_d128_a_one18.py`|verified|`test_stage2_d128_a_one18.py`|只接受single-candidate A bundle。|
|D128-02|复用D127 prepared plan和package materialization，固定18个before/after row pair|`stage2_d128_a_one18.py`|verified|`test_stage2_d128_a_one18.py`|只读取D127 sealed plan/row materialization。|
|D128-03|生成M0/M_DA/M_L92/M_JOINT；K1沿用等价alias|`stage2_d128_a_one18.py`|verified|`test_stage2_d128_a_one18.py`|四臂由既有D127 joint screen一次产生。|
|D128-04|prediction truth-free、独占写、hash绑定、query零fit/update/selection|`stage2_d128_a_one18.py`、`run_d128_a_one18.py`|verified|`test_stage2_d128_a_one18.py`|所有输出在truth-open前拒绝truth/role/quota字段。|
|D128-05|prediction封存后独立打开truth并构建D128 truth/formal reference|`stage2_d128_a_one18_scorer.py`、`score_d128_a_one18.py`、`build_d128_a_one18_truth_assets.py`|verified|`test_stage2_d128_a_one18_scorer.py`|D92 retry2文件只在durable truth-open验证之后读取。|
|D128-06|same-row指标与G1/G2/G3方向判据，不自主晋级|`stage2_d128_a_one18_scorer.py`|verified|`test_stage2_d128_a_one18_scorer.py`|输出判据，不包含promotion动作。|
|D128-07|不新增通用发布框架、不改D127现有模块|本追踪文件及新文件面|verified|`git diff --check`|仅新增D128专属文件。|

当前状态：8项聚焦测试、语法检查与3个CLI`--help`均已通过；尚待独立P0/P1复审和主agent提交。
