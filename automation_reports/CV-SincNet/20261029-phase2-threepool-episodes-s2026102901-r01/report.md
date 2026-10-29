# PAMF Phase2：三个互斥物理池、池内重采样、最小query150

- run_id：`20261029-phase2-threepool-episodes-s2026102901-r01`
- group_id：`phase2-threepool-episodes-r10`；类别：`diagnostic`；阶段：`Phase2-data-builder`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原6旧20新/7RX/4日554823条分3互斥池；每池8mixed+三pure各5组，K support与完整补集query；复用r09既有逐组信道，无身份实验。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## r10 最小预登记

用户已选择三池：原最小590条，四池K20最小127query，三池最小176query；要求至少150。原6旧类/原20新类/7RX/4日/sourceL6300/U56700隐藏TX/V27000不变。分层随机分三池且物理互斥、并集554823；每池综合8组+三pure各5组，每任务K1/5/10/20及N0/2/5/10/20。池内组间重采support/query及H，K内配对、全补集query。预计483bank/9660split/12760929次原有仿真观测，额外信道生成0；详细表见r10分配文档。

执行为N607既有Python，CWD独占release，入口python -u -m experiments.pamf_phase2_practical.three_pool_data --runtime runtime.json。完整命令/路径/owner/stop rule/seed及artifacts见experiment.json。单CPU进程、CUDA不可见、OMP/MKL/OpenBLAS1；r09健康8CPU数据工厂保留，r10以其原子完成状态增量消费已完成bank。每条received及匿名ID原样保留后拆入所属物理池，再重新构造pool内support任务。未准备好的输入15秒检查一次，上限24小时；不是重新叠加或重新上传数据。

一次N607 direct-only preflight及CPU/输入/输出路径检查通过：96CPU、当时约395GB可用内存、磁盘约2.17TB余量；r09已有48bank完成。新run/release不存在。首次输入不符、物理重叠/缺失、query不足150、输出碰撞、分片错误、上游失败或超时停止r10且保留partial，不停改r09或其他owner，不自动换类/缩量/覆盖。

10项聚焦检查PASS，涵盖穷尽互斥分层平衡、固定seed、非法输入、query下限、跨组固定物理归属、精确复用received/ID、A/B/C边界及全补集。唯一P0/P1审查发现并发读上游未写完JSON的P1，已改为等待上游既有原子progress完成状态；一次定点复审PASS，无未解决P0/P1，不修改健康上游代码。

不加载F/G，checkpoint smoke不适用；identity train/predict/score0，DURA缓存0。身份行预算理论上为r09三倍但本作业不启动身份队列。实际代码commit由full_publish从已推送GitHEAD写入runtime/submit，启动后回填；归档一次SHA比较、一次远端编译，随后独立PID/CWD/argv/日志/产物读回。最终完成须有partition_report、全部483capsule、catalog及builder_report；submitted不是complete。
