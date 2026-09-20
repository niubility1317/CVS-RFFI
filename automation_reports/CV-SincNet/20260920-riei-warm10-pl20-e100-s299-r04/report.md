# RIEI简单暖启动：100轮、有/无伪标签对照

状态：RUNNING，远端启动VERIFIED。两组均从零训练，seed299；固定E100 student为主结果。未加载历史checkpoint。

## 简要诊断

已完整解析旧r03的200轮、5800批日志，并从冻结概率独立重算最终准确率：source clean88.00%、source SG22.00%、target clean57.375%、target SG17.50%。stdout无异常。训练SG CE在E100–120均值仅0.0746，但同期source SG仅23.5–25%，支持固定强衰落训练样本拟合较好、SG泛化差的判断，不能证明唯一原因。

原信道是逐采样点Loo，log-amplitude标准差3（delta2=9）；从第一轮施加强扰动可能增加早期学习困难。原PL直到E100才启用，且SG视图接受量低于clean。这里只做用户指定的温和起步与提前PL，不据target分数选择checkpoint或调参。

## 两组配置

|组|训练轮数|无标签数据|伪标签|
|---|---:|---|---|
|noU|100|不索引U的IQ，不构造U loader，不做U前向|始终关闭|
|PL20|100|仅source U，E20起前向|E20含当轮启用，阈值0.9，系数0.65|

两组E1–10使用固定温和Loo：delta2=1，即log-amplitude标准差1；散射方差0.0049、逐点衰落、逐原始256段RMS归一化均保持原实现。E11–100恢复原固定强度SG。测试全程使用原强度固定数据，未弱化测试。温和增强seed340299；两组同一L顺序（299+epoch），U独立生成器（1299+epoch）。

RIEI原优化版保持不变：clean CE与SG CE系数1:1，RX CE系数1、MI系数1.2、IE系数-1.2、feature norm系数0.0001。CE/MI/IE仍sum、norm仍mean；Adam1e-4常数、L batch64、U batch256，交替更新heads/FED。无STAR E调度、动态增强、EMA或额外CFO/IQ。

输入先在原始两段256信号施加信道，再补零至512并分别FFT，最终布局time1/time2/freq1/freq2。不向已拼接FFT特征直接施加信道。

## 验证与证据

verification.json：本地CUDA真实RIEI更新通过；E19/E20的PL边界、无U步骤、禁止U IQ访问的loader守卫测试、暖增强布局/确定性及语法检查均通过。review.json：独立P0/P1审查无阻断问题。

diagnosis.json保存旧实验完整日志汇总与独立重算。新结果须等待各自COMPLETE.json及最终冻结预测独立评分；当前不宣称提升。本实验是目标结果已知后的单seed探索，不能作为盲测或显著性证据。

## 启动读回

- riei-warm10-no-unlabeled-s299-e100-r04：GPU5，PID241919，已记录416批；实际U加载数0。
- riei-warm10-pl20-s299-e100-r04：GPU7，PID241934，已记录269批；实际U加载数18000。

代码版本：29361aea1554ff69d218a23cafa252af02a8d35e。STAR本地仓库无remote，代码及证据同时镜像至CVS发布仓库。实际PID/CWD/argv见remote_inspect.json，加载数及增强边界见data_readback.json。
