# PAMF源端匿名跨接收机正对验证

## 预登记

Run `20261030-phase3-source-crossrx-neighbors-s2026092701-r01`，实际执行代码 `6eaf8d9a4f40f19834e317c50014836e2e96f689`。本轮只读sourceL6300/OE6134既有G10 MIX01；原F0 Tri-SIG-TIME SSL E50/11100/default6/232993冻结。P1完整单位fused160复用原F0teacher(code5aa041122396d48c49b8cb3d5912649bdc3217e1)，不是joint-OVA增量模型。缺P0特征由同F0对原received导出；逐ID和原F0/DURA/cachehandle相符。无newclean/重抽channel/Gload；L/OE为同原5RX/3day，physicalpool12434，不把双视图或重复计算当新样本。

完整方法/科学规则见`docs/PAMF_PHASE3_SOURCE_NEIGHBOR_PLAN_20261030.md`，矩阵唯一维护experiment.json。固定P1控制、PAIR唯一主：每其他RX且不同day最近邻，双向mutual；PAIR还要求P0/P1同条视图四个有向选择全部一致。FP64 cosine、opaqueID ties，不给候选函数y/K。仅L合法y校准最大可行距离τ，整ties处理；global+5RX+6TX各precision98%/incidentedges25。无可行τ=null/空边，不回退。全部L规则冻结后才挖掘/筛选OE，OE只有IDs/RXday/features，没有真实TX/K/配额。

主PAIR合法Lτ且L/OE总体节点覆盖各20%、每RX各10%、OEedges256，才允许另立下一源训练候选；控制不换主。直接边不传播组件硬标签，组件大小不是发现真实K，L拟合精度不代表OE真实精度。源V/novel/原Us/targetUQ/formalRX均0读，private truth输入不存在；AK/RU/未来AN/HOW N/A，DA/REG/神经更新0。完整targets v3不减，Phase3目标ACTIVE，本诊断不替代未来独立识别。

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-source-crossrx-neighbors-s2026092701-r01-release`；owner `codex-phase3-source-neighbors-20261030`。
- `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_source_neighbor_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-source-crossrx-neighbors-s2026092701-r01-release/runtime.json`。
- 独占root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-source-crossrx-neighbors-s2026092701-r01`，source GPU导出→CUDAempty/BLAS1 CPU挖掘；实际free4096MB、每卡最多新增2own，本轮最多1GPUworker。foreign不整卡排除、不停止/修改/重启。
- 每行没有optimizer/epoch；F新真实P0 12434+合成2，P1 feature12434精确复用，G0，原F0参数/缓冲不变。real-checkpoint syntheticG10 smoke先于cache加载，PASS直接继续，失败保留、不挖掘。
- 算法7tests/9sub PASS0.56s；IO8tests/15sub PASS2.91s，仅pytest cache权限warning，未重测或改权限；6AST/实际runtime。一次独立直接P0/P1 PASS；一次preflight仅句柄/祖先/文件存在/GPU，没有读取真实IQ/特征值/truth。单release归档比较/compile后启动，独立PID/CWD/argv/GPU/log增长；本轮无私有truth评分，不增加审批或hash链。
- 产物为features L/OE NPZ+smoke/complete、mining calibration/results/complete及四direct-pairNPZ、launch/pipeline/config/完整日志/observations；所有特征和原始日志私有不Git。非法角色/祖先、非有限、输出碰撞或运行不能闭合保留且不自动重试；科学无可用边记ANALYZED/NO_RELIABLE_SOURCE_NEIGHBORS，不作技术错误、不热改规则。UNKNOWN先核原PID/root。

目前仅预登记，不宣称启动或取得配对/识别结果。用户授权自主设计分工推进，额外设计审批/builder review/签名链NONBLOCKING。


## 实际完成与科学结论

VERIFIED source L6300/OE6134 G10 P0export/P1reuse; P1 candidate1358/1674 sameTX, PAIR22/25, no feasible L98%/25group rule; all final edges0; NO_RELIABLE_SOURCE_NEIGHBORS, no training/target/truth/G updates, fullgoalACTIVE.

详见 `docs/PAMF_PHASE3_SOURCE_NEIGHBOR_RESULTS_20261030.md`。
