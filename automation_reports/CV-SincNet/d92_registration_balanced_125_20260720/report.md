# D92注册任务均衡协方差125实验报告

## 实验登记

- 实验ID：`d92_registration_balanced_125_20260720`
- 日期：2026-07-20
- 操作者：Codex
- 状态：本地实现中，尚未启动N607
- 目标：在`p2_min_v1`协议下，对D62与D81的完整125实验结果进行联合诊断，开发不依赖场景、接收机、种子或具体类别标识的阶段二遗忘优化方法，并用同一125矩阵验证。
- 比较对象：D62与D81完整125矩阵；不以单一实验格点作为晋级依据。

## 假设与方法锁

D62与D81在注册类数增加后均出现系统性旧类遗忘。D65表明仅冻结旧类协方差能够改善旧类，却会显著损伤新类；D62/D81的全注册类共享协方差则更偏向数量更多的新类。D92固定执行：

1. 保持D81的地面压缩知识类内稳健中心变换；
2. 仅在注册后状态，将旧类支持与新类支持分别进行`auto-shrinkage`协方差估计；
3. 用固定`0.5Σ_old+0.5Σ_new`形成共享协方差；
4. 对全部已注册类使用同一个等先验LDA头独立决策；
5. K1或退化残差严格回退到既有单位协方差最近中心头。

固定等权来自项目对Stage2-B域适应与Stage2-C新类注册同等优先的任务定义，不进行场景、接收机、种子、新类数或类别级权重扫描。旧/新集合只来自注册状态支持清单；查询真值、查询角色、查询批次类数、类别配额和全局重分配均不可访问。

## 125矩阵

- 接收机：`20-1`,`3-19`,`7-14`,`7-7`,`8-8`
- 种子：`713102`至`713106`
- 切片：`K10/new5`,`K10/new10`,`K10/new20`,`K5/new20`,`K1/new20`
- 每个作业覆盖：`leo_clear_weak`,`leo_rain_weak`,`leo_low_elev_weak`
- 作业总数：125
- 评价指标：注册前旧类准确率、注册后旧类准确率、最差旧类准确率、新类准确率、`H_old_new`、遗忘、逐旧类准确率、场景/接收机/种子分层、覆盖/回退状态、资源与运行时。

## 版本与本地工作面

- Git工作树：`E:\type10-7\code\snapshots\d92_125wt`
- 分支：`codex/d92-registration-balanced-125`
- 起始提交：`d58bb056`
- 根目录`E:\type10-7`不是Git仓库；本报告将同步到根目录报告面，但版本化副本保存在上述Git工作树。
- 待改文件：D92核心方法、D92完整查询评估、行流水线候选分发、125启动器、125汇总器和窄测试。
- 本地环境：`conda activate ssr-gpu`

## N607启动信息

### 本地验证与版本

- Git提交：`94e39f529e926a5898c8e6cb92fe555d70fede07`。
- `ssr-gpu`中D92核心/集成测试7项通过；D81+D92联合回归22项通过。
- `ruff`未安装，未执行该可选检查；`py_compile`和数值/集成测试均通过。
- 远端隔离源码快照：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_source_snapshot_20260720`。
- 远端输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_20260720`。
- 远端日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_registration_balanced_125_20260720`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

### 同步文件与SHA256

|本地相对路径|远端相对路径|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d92_registration_balanced_covariance.py`|`cvsrffi/stage2_d92_registration_balanced_covariance.py`|`00ad4b7990a106ceffa89b4349ccf236df739d9fbc50213239f97ace079be934`|
|`code/cvsrffi/stage2_d92_query_evaluation.py`|`cvsrffi/stage2_d92_query_evaluation.py`|`5c4f4b86b4aba44fc9a4d8fe95b30428ae9a8c8b3da4cad3d8eeb86f6306356b`|
|`code/scripts/probe_d92_registration_balanced_covariance.py`|`scripts/probe_d92_registration_balanced_covariance.py`|`d8c1de5e8fbda769a03f266efd57f4651e89e3004b414f584fef3add3b8a9ae6`|
|`code/scripts/run_cvs_somph_diag_row_pipeline.py`|`scripts/run_cvs_somph_diag_row_pipeline.py`|`4556d84908f93aabfb60269449a19a43bcc0f8c2f46d915d1f7babc634797966`|
|`code/scripts/run_d92_125_stability.py`|`scripts/run_d92_125_stability.py`|`a59e1b1d2805b5a2a49efff9e7af5e6901c5a07a124ebb3aa36e7c9a79788d3c`|
|`code/scripts/summarize_d92_125_stability.py`|`scripts/summarize_d92_125_stability.py`|`7f69b03919ede05ec551c6f16c3645a19bb5763203567ae2f5a96e969adc0920`|

### 启动前资源现场

- 2026-07-20 14:08 CST直连预检通过；GPU0–4各已有2个D81行进程，GPU5–7各1个。
- 不终止、不修改其他任务；遵守每GPU最多2项实验，首批只在GPU5–7各增加1个D92分片，其余5个分片在容量释放后补齐。
- 每个D92子进程CPU线程上限固定为2，设备参数为`cuda:0`并通过`CUDA_VISIBLE_DEVICES`绑定物理GPU。
- 精确PID、命令与清单SHA在实际启动后补充。

## 成功标准与风险

- 方法晋级必须看完整125矩阵的联合指标，不能用单个格点或边际最大值替代。
- 主要目标是降低注册后遗忘，同时不以明显牺牲新类准确率为代价；严格目标以活动目标文件为准。
- 已知风险：K1无组内残差，D92会严格回退，因此K1可能不改善；旧/新协方差的尺度差异可能使固定等权仍然偏置；D62安全门可能使部分D92分量回退。
- 若125实验完成但未满足目标，结论应为完整诊断阴性，不得宣传为性能晋级。

## 完成后结果表

待125实验完成后补充逐候选/逐实验同一行指标、最佳联合排序、分层退化、异常、资源和下一轮建议。
