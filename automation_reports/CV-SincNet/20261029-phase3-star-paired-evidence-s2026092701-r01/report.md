# 星地接收与恢复联合拒识证据：预登记

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
