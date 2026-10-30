# DURA源域GPU训练步骤耗时诊断

- run_id：`20261030-diagnostic-DURA-training-profile-s2026092701-r01`
- group_id：`phase1-dura-training-step-profile`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FrozenlegalfastLRsourceE20;B256same-source individual/buffered metrics 3x3alternating;4baselineprofiledsteps;0optimizer/F/TX/V/target

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

## 完整诊断结果

VERIFIED/ANALYZED_SOURCE_ONLY。一次独立P0/P1审查PASS，既有checkpoint来源负测1 PASS；一次N607 preflight、release archive本地/远端SHA匹配、compile通过。执行commit a5a65887ae34e43d5f2edc0764981db86ef4ecb6，CPUcontroller3657300/GPU4 worker3657396。首轮回查前该短诊断已结束：两进程当前均不存在，独立回收worker.json、实际CUPTI PID3657396收集完成日志、正CUDA耗时event和complete.json，证明实际GPU执行及结束；未声称直接观测到运行期间nvidia-smi。实际CWD/argv/GPU见worker产物，输出日志从无变为320bytes。

B256同源epoch20、2warm＋3轮交替×每路径3timed＋4原路径profiled，0optimizer/TX/F/V/target。全部裁剪后参数梯度/损失在预登记rtol2e-5/atol2e-6内一致，模型所有权重/缓冲区不变。完整9次墙钟和32bit source目标已独立复算中位数。

逐项统计路径中位131.502ms，合并统计137.562ms；三轮各自中位也均未变快，整体约慢4.61%。共享负载下不能作严格因果速度比，但没有采用该优化的证据，故不修改现有训练器。profile统计回传CPU总耗时1.572ms/4步，source准备19.368ms/4步。

原路径有15次linear1d上采样/step。4profiled步的aten::upsample_linear1d CUDA self60.283ms、其backward79.968ms，均为突出event；卷积反向55.946ms。原始profiler含父子嵌套和异步线程，不能把op与kernel重复相加或据此计算完整墙钟占比；仅用它定位下一源端实现候选，不声称已加速。profile开销与未profile计时分开，结果不代表epoch或完整训练成本。

原始24步记录、完整profile event、日志与状态留在local_artifacts/remote原root；Git仅公开汇总。既无权重更新，也无新的识别候选，不启动target测试。下一零更新诊断只替换Core内12/15次二倍线性插值，检验相同半像素边界、前向和全参数反向以及同卡时延；不会改当前健康训练。
