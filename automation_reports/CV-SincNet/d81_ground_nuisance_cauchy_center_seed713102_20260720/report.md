# D81地面干扰谱Cauchy中心：seed713102独立确认

## 实验登记

- 实验ID：`d81_ground_nuisance_cauchy_center_seed713102_20260720`
- 登记时间：2026-07-20 04:55 HKT
- 操作者：Codex
- 目标：在`rx20-1/K10/new5/3场景×5 outer folds`上，用独立确认种子`713102`复验D81；判断seed713101的联合改善是否具有种子稳定性。
- 方法锁：`ground_nuisance_cauchy_center`；代码提交`2f6a26d3`，首种子完成报告提交`930481d5`。
- 对照：同一seed、同一capsule上的D62 target-support-only基线；同时列出D81 seed713101同row结果。
- 假设：84个地面int8类中心只估计类中心化跨域干扰谱，并用固定一步Cauchy权重稳健估计target类中心，可降低support中心漂移噪声；地面原型不进入query评分，查询附加计算为0。

## 数据与协议

- `protocol_schema=p2_min_v1`；固定单次`LEO_weak`已接收IQ，support-only适配，query逐样本独立评分。
- receiver=`20-1`，seed=`713102`，K=`10`，实际每类/场景support=`8`，seen-new=`5`，scenes=`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，outer folds=`0..4`。
- 数据状态：D18母缓存30/30 cells已`VALIDATED_ONCE`；本次复用`rx_20_1_seed_713102`，不重建、不重验、不改变物理ID、scenario assignment、support/query split或schema。
- 禁止项：无clean/source访问；无query truth、role Oracle、类配额、真实batch类数、global reassignment或query-dependent batch optimization。
- 地面组件：84个有效int8聚合单元；NPZ SHA256=`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；当前签名状态仍为`UNVERIFIED`，因此结果只能是开发诊断证据。

## 版本与本地验证

|项目|状态|
|---|---|
|Git承载面|`E:\type10-7\github_publish\CVS-RFFI-repo`，主分支存在大量不相关工作树改动，严格按路径提交|
|隔离worktree|`E:\type10-7\code\snapshots\d81wt`，登记前HEAD=`930481d5`|
|D81实现|`code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py`，SHA256=`44111f8d7ecd0ffcfbd887c09468a167e4e1134bad3c2798bd7f0f5f89c3dc7a`|
|D81 runner|`code/scripts/probe_d81_ground_nuisance_cauchy_center.py`，SHA256=`85baac449d2cd1c5b21bff63ba9b01fe95bb2025fcdfa8ee3127ae41a5e99e82`|
|已完成验证|D81 unit+synthetic D62 stack 11/11 PASS；D62/D80/D81相邻链30/30 PASS；seed713101真实ground smoke PASS|

## N607封装与运行计划

- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/source`。
- 复用缓存：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713102`。
- 新输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_ground_nuisance_cauchy_center_seed713102_20260720`；不得覆盖D18或seed713101产物。
- 环境：N607`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；本地`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- GPU：authority/capsule构建为CPU；D81确认计划本地执行，不占N607 GPU。若改为N607执行，先再次记录GPU占用且不超过每GPU两条训练进程。
- 私钥：仅本机`authority_signing_private_ed25519.pem`使用；绝不上传N607。
- 过程：远端用D18精确source复核seed713102真实缓存并生成unsigned authority；本地Ed25519签名；远端生成authority bundle和exact-K row pair；本地生成path-free runtime authorization并执行D81。
- PID：短时前台封装命令，无长期PID；若任何步骤改用后台命令，将在本节补记PID、GPU和log。
- 计划log：本报告目录下`logs/*.stdout.log`、`logs/*.stderr.log`；远端封装log位于新输出根`logs/`。
- 预期输出：独立seed713102 authority/capsule/runtime authorization、105-row完整训练日志、30 target rows、性能summary、资源/量化/协议审计和`RECEIPT.json`。

## 成功、停止与升级条件

- 确认成功：seed713102对matched D62满足D81预注册联合门——B/A/N/H/F/J、三场景、逐类floor、mean-row floor和三类confusion均不回退，且A/H/F或old-to-new confusion至少一项严格改善。
- 失败：任何联合门回退、收益消失或只由不稳定单样本翻转构成，均记为确认失败并详细报告，不启动125矩阵。
- 协议失败：ground/query访问、hash drift、signature/seal、support/query不交或单观测约束任一失败即fail-closed。
- 升级边界：只有独立种子确认成功，且ground正式签名问题解决后，才考虑扩大到多receiver/多seed；当前不启动125实验。

## 初始远端证据

- 2026-07-20 04:48 HKT direct N607 preflight PASS：server time、project root和8张RTX3090可见。
- 8张GPU均约10MiB且无活动实验进程；seed713102三个场景NPZ与`cache_set.json`存在。
- 已确认seed713102没有现成capsule；本次生成独立authority，绝不复用seed713101的seed-specific签名。

## 完成结果

待实验完成后补充：完整同row性能表、逐场景/逐类/混淆、量化一致性、训练日志统计、资源、异常、缺陷、最终判定与下一实验。

## Capsule完成与确认入口修复

- N607 unsigned authority成功：lock SHA256=`7d23365047a6ca0d6da885170b6c8fb0056586cf940619cf6450e1f893b12365`；build receipt SHA256=`ff29c1e3c98bea752a6e20e8a406df1610b2553421f430af5a279f489694c01d`。
- 本地Ed25519 envelope SHA256=`09d97a8f70fcb40cc7540bc4941c9756ba961532fca366bf8d44a060aacf2327`；私钥未上传。
- N607 authority commit=`59e6a6e0afcd7b519829e8e1a6ccc25ae43d07acd05815013542ee4cf6cc1c79`；exact-K row pair的同场景support/query不交、跨场景物理ID不交和before/after旧类复用审计均PASS。
- before/after enrollment seal分别为`6c2961f6f29f74ec885a6279529c813c462209e8b2db4d8e743bcb9e5fb2754f`、`ee0050b28ed3f10f90c7a19677896cab02f1ceae50777884263643306d02c454`；runtime closure=`b0b7f2c2f87e66ecbeca99779688461e7161877271dd0195e0bcf2b95cb9606f`。
- 第一次D81确认启动在读取query前fail-closed：继承执行器只允许开发seed713101，错误为`D42 preregistered development cell must be receiver 20-1, seed 713101, K10, new5`；退出码1、无输出目录、无性能数据。
- 最小修复只在D81 probe增加显式`--d81-confirmation-seed`入口；允许集合固定为`713102..713106`，并同时核验before/after receiver、seed、K、旧类前缀和new5。未提供该参数时仍使用原开发单元锁，未修改D42通用执行器。
- 修复后D81 probe SHA256=`f18dd80dd2f38312dde41e07120f43dc9cd2b78afa82b54c65fe08236e513817`；专项测试13/13 PASS，相邻D62/D80/D81链32/32 PASS，`py_compile`与`git diff --check`PASS。
- 重跑命令在原锁定命令基础上只新增`--d81-confirmation-seed 713102`；其余capsule、ground、runtime和query边界不变。
