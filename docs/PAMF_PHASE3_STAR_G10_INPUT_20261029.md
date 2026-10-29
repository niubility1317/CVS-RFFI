# Phase3 统一星地输入与 G10 开发数据（2026-10-29）

用户明确“脱离phase1后，后续接触的样本都是叠加星地信道的”，并指定参考 [评估 MILKR 的无标签适配](thread://01a18197-50fe-7f93-bd50-7f98d3f59cff?hostId=local)。已通过 `read_thread` 读取该任务的实际说明，再核对 r09/r10 文档、混合配置以及该任务已提交的 `data.observe`/`geometry` 实现。本文件记录新的输入约束和首个小数据作业；不声明新 G10 方法性能已测定。

## 后续输入范围

Phase1 原权重和源数据身份保持。之后新进入 Phase3 模型的源端 replay/辅助训练/校准 IQ、目标无标签适应/发现、匿名注册和未来 query 都必须是星地观测。原始地面 IQ 只用于离线生成 received，不供预测器逐条参考；不再新增 Phase3 clean 适应或注册路线。

P0 为一份固定星地 received；P1 为这同一份 received 经固定 DURA v1.0/R69 的恢复结果。P0/P1 不是 clean/star 两组。各方法和模型 seed 共享当前 episode 的 received；新 episode 另抽信道，不覆盖历史文件。Phase3 U/Q 角色互斥与零目标 TX 标签保持，不引入 Phase2 K-shot、真实未知 K 或 query 转训练规则。

此前 Phase3 三景压力测试采用 2462 MHz 和独立 receiver seed，场景在每类内等量分配。它的结果继续按旧条件记录；本 G10 主条件使用 2450 MHz、receiver seed2027 与逐记录概率抽样，不继承旧性能。

## 星地叠加设置

实际代码参考：`E:/type10-7/code/worktrees/standard_ce_baselines_20261010/experiments/pamf_phase2_practical/data.py`，最近相关提交 `71f1428c8ac85695ebffb2efd791fe34061b325b`。星地基础实现为 `code/leo_practical/{channel,residual}.py`，版本 `leo_practical_v3_six_scenarios_20260919`。原始配置为 [MILKR G10 六分量设置](../configs/pamf_phase2_practical_mixture_20261028.json)；其 DESIGN_ONLY 等旧状态文字不覆盖 r09/r10 已实际执行的观测代码。

- 高度 600 km、采样率 25 MHz、载频 2.45 GHz；residual、post_sync、no additional EQ；reference SNR 30 dB、reflect。
- 原 float64[256,2] 只转置为[2,256]；按参考实现先转 FP32、再在复数信道入口做一次 unit RMS。输出存 FP32，不再逐条 RMS，也不重新均衡 WiSig 原 eq=1 数据。
- receiver session=`rx:<RX>/day:<date>`，receiver seed=2027，独立于 TX/旧新角色/UQ/场景。
- 综合场景先抽外层六分量，再在对应互斥仰角带内均匀采样 cos(psi)，计算实际斜距和带正负号的圆轨道径向速度。内层状态仍用原生仰角/环境的 LOS/PARTIAL/BLOCKED 模型。

| 外层场景 | G10 概率 | MIX 仰角带 |
| --- | ---: | --- |
| high suburban | 0.04369141011349172 | 45°–80° |
| mid suburban | 0.07317984658235889 | 30°–45° |
| low suburban | 0.38312874330414937 | 10°–30° |
| high urban | 0.04369141011349172 | 45°–80° |
| mid urban | 0.07317984658235889 | 30°–45° |
| low urban | 0.38312874330414937 | 10°–30° |

两种环境各0.5；这些是覆盖面积参考概率，不是实测卫星任务占比。若运行指定单景，low suburban/mid urban/low urban 使用原生均匀角度10°–30°/20°–45°/10°–30°；单景 mid urban 的20°下界不用于 MIX 的 mid 档。

## 首个 G10 开发数据

数据 run=`20261029-phase3-star-g10-pool-rx24-1-r01`，配置为 [新 G10 数据配置](../configs/pamf_phase3_star_g10_pool_20261029.json)。仅制作一个 MIX01，scene seed2026103001/channel seed2026104001，与参考第一套的种子约定一致；当前不铺满8×三池或新模型实验。

复用旧 RX24-1 builder 已固定的相同物理记录与角色，不按 IQ/模型分数选新记录。新 builder 仅读旧 manifest/IDs/私有 canonical 映射和两个既有验证通过的 Full 成员；不读取旧 received、原 truth NPZ 或评分来选择。Full 的158成员验证复用，不全量重扫。

| 角色 | 日期 | builder/scorer 私有设备集合 | 原物理记录 |
| --- | --- | --- | ---: |
| 无标签 U | 2021-03-08 | 旧6＋19-15/19-16 | 384 |
| 独立 future Q | 2021-03-15 | 上述8＋后续7-1 | 864 |

共1248条原物理记录；本次只增加一个新固定星地实现，不称新增1248个独立物理样本。旧6为14-10/14-7/20-15/20-19/6-15/8-20。source五RX与开发RX/正式RX24-18隔离；本次正式池读取0。设备此前source开发暴露背景保留，不称新TX盲测。

公开数据每角色仅manifest.json/ids.npy/received.npy，仍为phase3_min_v1；没有TX、真实未知数、类别配额、scene、geometry或clean参考。新的opaque ID带本run/episode，防止与旧信道输出混用。真值和实际场景/仰角/seed只存离线builder/scorer侧。方法从U自行发现未知数，Q只逐条预测。

这次数据作业不加载F/G、训练、预测或评分，不改变Phase1/2任务。实际构建状态以run登记、PID/日志与complete产物读回为准；完整性能候选随后另立run，并使用星地≥90%、已知FRR≤5%/同输入open损失≤2pp、未来新类macro≥80%及注册旧类保持等全部目标。

## 实际构建结果

**VERIFIED／VALIDATED_ONCE／ANALYZED_DATA_ONLY**。实际代码/config `3f24f8016f5586f096bbaa8d83f221b397dbbeb8`，U384/Q864，1248个原物理记录各增加一份本MIX01固定星地实现；仅两已索引Full成员，未新增独立物理样本。旧U/Q角色保持，旧received/truth/scene/score读取均0；新F/G加载、训练、预测、评分、正式RX24-18读取和GPU新增均0。

| 角色 | 实际场景 | 记录数 | 实际仰角°范围 |
| --- | --- | ---: | --- |
| discovery | practical_high | 17 | 45.7103 至 76.9954 |
| discovery | practical_mid | 27 | 30.7488 至 43.6544 |
| discovery | practical_low_suburban | 158 | 10.0764 至 29.6692 |
| discovery | practical_high_urban | 20 | 45.4529 至 76.2168 |
| discovery | practical_mid_urban | 32 | 31.3383 至 44.7697 |
| discovery | practical_low_urban | 130 | 10.0724 至 29.9985 |
| evaluation | practical_high | 39 | 45.1953 至 78.1206 |
| evaluation | practical_mid | 67 | 30.0146 至 44.7757 |
| evaluation | practical_low_suburban | 320 | 10.0422 至 29.6956 |
| evaluation | practical_high_urban | 54 | 45.2506 至 77.8500 |
| evaluation | practical_mid_urban | 60 | 30.5614 至 44.9584 |
| evaluation | practical_low_urban | 324 | 10.0629 至 29.8799 |

六景按参考G10概率逐记录抽样，有限样本计数无需等于配额，不按真实类别或角色重平衡。公开每角色精确三文件、phase3_min_v1；原始canonical、场景/仰角/seed和truth留私有builder/scorer侧。

discovery capsule=`p3dev-star-g10-6e1292b62ab09286ebf1a0df`，split=`83c5cc1fe2f2a1512dbc1287`。

evaluation capsule=`p3dev-star-g10-71a5a0dec23ac6ccc3b479c2`，split=`05de65cd4818f3e5e33ced5c`。

完整读取本次11行stdout，无异常；已独立捕获真实子进程PID/命令与CLI的CWD，日志增长与终态artifact由build_observations/build_post_state记录；不把启动返回值当完成。此次只证明输入数据就绪，新的G10拒识/适应/注册性能尚未测定，Phase3目标保持ACTIVE。原clean与三等量压力景成绩不迁移到本设置。

证据：`local_artifacts/pamf_phase3_star_g10_pool_20261029_r01/{pool/complete.json,pool/builder_report.json,analysis_summary.json,build.log,build_observations.jsonl,build_post_state.json}`；无IQ/私有truth/原始日志入Git。
