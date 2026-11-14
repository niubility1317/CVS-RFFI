# Phase 2 clean 自动评分

一个独立 CPU 评分 owner 等待预先确定的三池、四模型 seed、全部任务及 K/N：PAMF、六个原生域适应、四个原生注册、六个域适应与三个注册核心的组合。收集已完成且有合法 clean 数据绑定的历史 pilot、接续 run 和后续 clean 预测；已知作废 pilot 与失败的早期整队列不纳入。未完成或隔离任务不能视为完整。按方法、阶段、输入路径、K、N 逐行核对；重复结果冲突则停止，不以平均掩盖 K/N。

全部覆盖后先验证每份 prediction 的任务、数据口径、行和文件，再开启独立 truth-last 评分；最多 4 个 CPU 进程，无目标反馈。输出独占 `runs/20261115-phase2-clean-auto-score-m4-r01`，每个任务保留 natural 和 common-K，PAMF 两种路径分开；真实 B 状态不同的结果不可声称共享 B。执行 `python -m experiments.pamf_phase2_practical.clean_auto_score --preparations <project>/preparations --output <project>/runs/20261115-phase2-clean-auto-score-m4-r01 --interval 300 --workers 4`。

当前 ISSL、若干混合组和 MoPC 均未完成，且 MoPC 已发生真实非有限；观察器应保持 WAITING，不打开 truth，不重复预测。发生预测完整性或评分错误保留日志与输出并停下，不能把部分结果冒充全量评分。
