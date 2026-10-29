# 星地接收与恢复联合拒识证据：预登记

当前登记状态：ANALYZED_SOURCE_ONLY；两stage COMPLETE，实测 NO_PROMOTION。以下预登记正文保留，结果见末节。

run `20261029-phase3-star-paired-evidence-s2026092701-r01`，发布前登记，不是实测成功；完整Phase3目标ACTIVE，适用用户最新v3双90%/同输入原F0闭集到开放损失2pp、宏A_K/R_U/H。

实际code `bcc873cd9f9273133d74b5bc591c947b37ce16e1`；30项相关测试与56subtests PASS，一次独立下一实跑P0/P1审查PASS，本地runtime validate和一次N607 preflight PASS。实际原F0无query合成G10 smoke是预测第一步，PASS立即继续，无token/receipt。

方法/矩阵/预算/源晋级/停止规则见[最小设计](../../../docs/PAMF_PHASE3_STAR_PAIRED_PLAN_20261029.md)。原F0/defaulthead冻结，身份始终P1；P0仅拒识辅助，pair200=(P1consensus100,P0consensus100)。匹配MLP3233、相同初始权重，两arm各1000CPU全批更新，同轨迹step200/1000预定快照共4状态，总2000实际updates。P1-only后100在拟合/归一化/预测前置零，无P0泄漏。只主paired1000可晋级，控制不可替换。

L6300/OE6134拟合，knownV27000只42组源阈值，novel3150仅全部状态/阈值冻结后预测与独立truth-last。均同五源RX/三日，novel3已历史开发暴露，不是未见RX target或新TX盲测。复用已完成G10 P0/P1 cache，无raw/clean/新G/真实信道生成或新独立物理记录。target/dev/formal U/Q为0。

源判定：42known组宏CK≥90%、原F0P1closed−open≤2pp，unknownoverall/5RX/6scene≥90%，无H硬门槛；缺必需类N/A。paired−P1控制比较输入，1000−200比较预算；每项报告同Q配对净数与宏主指标。只有源通过才获下一独立冻结小规模跨RX验证资格，不能声明完整Phase3成功。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_paired_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-paired-evidence-s2026092701-r01-release/runtime.json`。
环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-paired-evidence-s2026092701-r01-release`。
run/output：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-paired-evidence-s2026092701-r01`；logs：同root的`logs/`；local：`E:/type10-7/local_artifacts/pamf_phase3_star_paired_20261029_r01`。
cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`，只读原四角色；source novel truth只在独立CPU scorer打开。
GPU：0..7实际free≥4096MB一张，单预测child，本任务新增每卡≤2，不停其他owner。实际PID/CWD/argv/GPU/log增长启动后独立观察；未捕获实时GPU采样时如实说明，不伪造或为证明监控重跑。

产物root为prediction/complete.json、geometry/P0/P1、readouts.npz/readouts.json、thresholds.npy、calibration/complete.json、predict/known_predictions.npz和novel_predictions.npz；独立score/results.json和complete.json。同row完整分析后代码/报告/登记Git提交push并远端OID读回，IQ/权重/原始日志和预测不提交。

技术非法输入/继承/非有限/输出碰撞/无法合法预测评分则保留partial，不盲重发；科学低分仍完成全部固定预测/评分，不调阈值、选控制、扩seed或打开target。REJECTED_EXTRA_GATE：无独立设计审批、重复数据验证、file/member hash、receipt或额外报告审查发布门槛。一个代码commit、一份archive一次local/remote SHA与一次compile、一次独立审查和实际smoke按项目最小流程完成。

## 2026-10-29 G10 源端 paired evidence 实测结果

`20261029-phase3-star-paired-evidence-s2026092701-r01` 两stage完成；唯一主paired1000 **NO_PROMOTION**，完整Phase3目标保持 **ACTIVE**。

| 方法 | A_K% | R_U% | H-score% | C_K% | F0损pp | 源gate |
| --- | --- | --- | --- | --- | --- | --- |
| energy | 90.1444 | 10.2540 | 18.4134 | 97.7852 | 0.8370 | False |
| P1-200 | 90.2222 | 44.8571 | 59.9220 | 98.1259 | 0.7593 | False |
| paired-200 | 90.1815 | 41.8095 | 57.1318 | 98.1222 | 0.8000 | False |
| P1-1000 | 90.2111 | 43.9048 | 59.0638 | 98.2185 | 0.7704 | False |
| paired-1000 | 90.3593 | 43.7143 | 58.9227 | 98.4778 | 0.6222 | False |

v3 known42组CK≥90%/同输入原F0 P1 closed−AK≤2pp全部通过，unknown overall/5RX/6scene共12组均未达90%。AN/HOW=N/A，target/适应/发现/注册访问0。
同初始化3233参数、两条各1000CPU全批轨迹总2000更新；200快照不新增优化。主−同预算P1-only：ΔAK+0.1481pp/ΔRU−0.1905pp/ΔH−0.1411pp，净正确+40/未知拒识−6；AUROC+0.8848pp/OSCR+0.7921pp，排序改善不等于运行点过线。
主−energy：ΔAK+0.2148pp/ΔRU+33.4603pp/ΔH+40.5093pp，净正确+58/未知拒识+1054；paired1000−paired200净正确+48/未知拒识+60。五方法和其余预算/输入配对均如实报告，不按分换主或扫描阈值。
复用G10缓存42584既有物理记录，P0/P1双路85168 F前向，F/G/head更新0、Gforward0；21OE额外membership源监督，V额外known标签仅校准，novel历史暴露且同源RX。主部署共享F+G+MLP503177参数，不是两个F。
独立live确认driver2925797/source2925886 CWD/argv/CUDA2，GPU UUID4fb…388MB离散样本非peak；source日志11674→21345B，终态owned/live空、foreign0。pipeline35.992259s是整过程时间，不是逐包延迟。
完整五方法/42组/逐TX/五项配对净数、2000loss与四快照、域权重/状态/观察边界见 [`PAMF_PHASE3_STAR_PAIRED_RESULTS_20261029.md`](../../../docs/PAMF_PHASE3_STAR_PAIRED_RESULTS_20261029.md)。
