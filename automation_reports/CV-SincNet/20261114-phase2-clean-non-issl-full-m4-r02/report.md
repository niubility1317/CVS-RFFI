# Phase2 clean 其他方法优先

用户要求先运行其他方法。父队列 `20261114-phase2-clean-ready-full-m4-r01` 已停止新增派发，12 个在途 ISSL worker 已精确停止，日志、部分预测和父输出全部保留；父状态 FAILED 来自主动停止，不视为 ISSL 科学故障。其余方法未派发，依据原任务清单、日志和输出逐项检查并安全接续。

后继只接未开始的 PAMF、6 种原生域适应和 LoRa（15416 作业），并使用已完成且伪标签正常接纳的四个修复版 Orthogonal 源模型增加 1927 个原生 Orthogonal 作业，共 17343。ISSL 的 1927 项暂缓，12 个已启动 ISSL 的 partial 不重发。先前 5 对 capsule/seed 的 pilot 其余 K/N 点、MoPC 和交叉对照仍待单独接续，不能把本队列称全部实验。

使用 GPU 1、2、3，每卡本任务最多 5 个进程，保留每卡 6000 MiB 启动预算，避让其他 owner。运行根 `runs/20261114-phase2-clean-non-issl-full-m4-r02`，固定代码以本报告所属提交为准。生成计划先为父 run 原子保留唯一后继，再启动 clean_scheduler；真实源权重 smoke、预测固定、故障停止派发和排空保持。完整预登记范围未固定前，独立 truth-last 评分不启动。
