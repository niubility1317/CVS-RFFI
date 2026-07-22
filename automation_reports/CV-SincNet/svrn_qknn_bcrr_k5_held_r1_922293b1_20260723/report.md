# SVRN-qKNN-BCRR/r2 K5 held发布报告

- run_id:`svrn_qknn_bcrr_k5_held_r1_922293b1_20260723`
- candidate/revision:`SVRN-qKNN-BCRR/r2`
- lifecycle:`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- operator:主agent`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- method Git commit:`922293b1cc2e15a2f595fc124074bae217ae427e`
- design-freeze Git commit:`407144dd714270bd8305595176dfcf921246b75d`
- protocol:`p2_min_v1 / VALIDATED_ONCE`；复用GEOFF/r8，不重验数据

## 1.目标与假设

在同一receiver=`1-1`、K=5、6个pseudo-new×3个`leo_*_weak`场景的18个slice上产生72个同row分数。M0为基础qKNN；M_DA为SVRN＋qKNN；M_OTHER为qKNN＋连续BCRR；M_JOINT为SVRN＋qKNN＋BCRR。假设SVRN改变目标域邻居/argmax，BCRR只修复剩余类间边界且不改变qKNN bank/邻居，联合后mean`I_syn(H)>0`并跨slice/scene成立。

matched reference为同一GEOFF/r8 archive和coverage上的基础qKNN；历史CID-BPP仅作诊断对照，不复用其packet、prediction或score。

## 2.冻结机制与停止门

- `M0=Q_raw`；`M_DA=Q_svrn`；`M_OTHER=F_raw`；`M_JOINT=F_svrn`。
- `F=(1-ω)N(Q)+ωN(B)`，`0≤ω≤0.5`；`ω_q=floor(254ω*)/254`。
- raw/SVRN独立使用同物理ID LOO双向cross-view逐类CE安全集；support和BCR权重均走正式qint8＋fp16 scale解码路径。
- query不更新η、ω、bank、温度、回退或任何状态；每个query独立面对全部注册类。
- 必须同时满足：DA净正确>0且old/new净变化非负；OTHER独立正收益；单组件保护全部P↑/P↓；JOINT mean H严格高于两单臂；mean`I_syn>0`；严格正协同≥9/18；正scene均值≥2/3。
- 任一门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得运行125。

## 3.本地闭合

|项目|结果|
|---|---|
|专项`py_compile`|PASS|
|专项测试|8 passed|
|相邻qKNN/R2A/CID回归|40 passed；仅pytest临时目录清理PermissionError的P2环境噪声|
|真实r8 support-only smoke|PASS；receiver=`1-1`、scene=`leo_clear_weak`、C5 support=25、query/truth fit=0、canonical IDs=25/25、wire=30,060B、optimizer step=0|
|独立review终审|`MERGE / P0=0 / P1=0`|
|非阻塞P2|未直接覆盖K10；未显式记录前向次数|

真实smoke的`η=ω_raw=ω_svrn=0`仅验证identity回退与wire闭合，不是性能结果。

## 4.发布源与输入绑定

|artifact|SHA256/说明|
|---|---|
|源码ZIP|`E:/type10-7/code/snapshots/svrn_qknn_bcrr_k5_held_r1_922293b1_20260723/source_922293b1.zip`|
|ZIP SHA256/大小/entries|`0ded7b8bdc3f70163dc38bdaa89e93f298e9beaea83f4190ccc04ad472176ab4`；33,311,517B；4,475|
|ZIP内core|`260d53b0e92732939446a5b0385d48fae94d60eb7d791e75ac7c54b925727f7e`|
|ZIP内held|`986ef162a725e194f7bb97ca347a418cd4e7c593699bb3a09260b87d098632ab`|
|ZIP内test|`f43d0047599faf4b608fa9f4301113fa137483baeb373df4e8032ac8e5d1409b`|
|wrapper SHA256|`4ba0671ee7ca62db63656f308e0ffbc841345c45f1b6319449cdb0624324cf14`|
|r8 parity|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|r8 archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|r8 manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|r8 coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|

## 5.N607冻结执行合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_held_r1_922293b1_20260723`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source root:`<remote-root>/source_922293b1`
- r8 root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`
- 唯一启动:`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &`
- retry:`NO`；不得远端编辑、调参、重建数据、复用旧run、kill/restart或启动125。
- 预期artifact:`packet.json`、`truth.json`、`query.npz`、`prediction.json`、`score.json`、`sha256sums.txt`、`complete.marker`、PID、exit和完整log。
- 预期数量:prediction slice=18；score row=72；arm=`M0/M_DA/M_OTHER/M_JOINT`。

runner必须完成direct preflight、run root不存在/GPU/进程/磁盘检查、ZIP/wrapper/source/r8 SHA、单根布局、远端`py_compile`/import、`bash -n`，然后只启动一次并用短连接监控、完整回收artifact。无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 6.完成后分析要求

必须独立复算18×4同row指标、logits→argmax、truth/prediction绑定和SHA；报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类、scene、K、receiver、η/ω、coverage、邻居变化、量化margin、state、MAC、mean/P95、VRAM以及`I_syn`。不得拼接不同row极值或把进程完成当作性能成功。
