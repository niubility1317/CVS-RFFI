# 同一源域受损IQ：CSIL、MoPC-HR与当前PAMF的冻结互补检验

- run_id：`20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r01`
- group_id：`phase1-same-observation-external-teacher`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用P79缓存和AS30默认头预测；固定历史seed01/final200外部模型读取相同IQ，先证伪教师资格。0训练、0新增信道、0目标访问。

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
# 最小预登记：相同受损观测的跨模型互补

本轮选择不同于 U 路由和 HQ-labelCE 的路线。先比较固定历史 CSIL、MoPC-HR 与当前 PAMF 在完全相同源域 IQ 上的判断，只有源证据支持才设计训练期类别监督。原 CVS、joint160、默认 cosine30 头及部署结构不变。已有 T2/半 CE、前缀屏蔽和质量加权负结果不重开。

矩阵、六类 seed 与路径以 experiment.json 为准。每个外部模型固定 seed2026092701、从零 final200/9800 更新；PAMF 固定原 P43 As30/6660 更新。成本不同，不能据此声称匹配训练下的架构优势。冻结历史头，无新增头、拟合或 BatchNorm 更新。

复用源 V1800、两次既有 draw、clean 与 lowSuburban/midUrban/lowUrban full IQ；P79 同包 AS30 logits 直接复用。两份历史模型的 JSON 来源已核对：L6300/U56700/V27000 全部实际物理 role_ids 精确一致、无祖先、固定末轮、无 target 训练或选模。运行时在真实 checkpoint smoke 前再次核对 checkpoint 内 payload 与既有元数据。数据没有变化，不重建或重验缓存。

教师资格分别判断，不按历史正式分数挑 seed：每个 draw 上，low 相对 AS30 至少 +1 个百分点，mid/low 均值至少 +0.5 个百分点，两 draw 均须满足。否则关闭该固定教师，不用温度、权重或目标成绩救援。资格通过只支持设计下一项有匹配控制的训练实验，不代表目标已完成。

新增预算共 25204 次样本前向（每模型12600源观测＋2真实CP合成smoke），0更新、0新增IQ、0query。source cache 加载可能反序列化未使用的 L；只读取 V 角色，不迭代 U 或访问 U 真值。oracle union 是冻结模型互补的描述，不是可部署准确率。

命令：CVS-RFFI/python -u experiments/cvs_sameiq_experts/runner.py --project PROJECT --device cuda:0。CWD=PROJECT/releases/cvs_FcrossRX_sameIQ_experts_20261019_r01；唯一 owner=codex/root/sameIQ-experts；单GPU0进程、CPU2、FP32/TF32off，闲置GPU预检，否则等待。实际代码身份写 release_commit.txt；一个归档、一次本地/远端SHA比较、一次远端compile。首步真实CP无query smoke通过即继续。

仅契约/类别映射/来源/形状/非有限/冻结状态/输出冲突等技术失败中止本run，保留产物，无自动重试。低性能完成全部预登记读出；不触碰健康任务。无目标测试或独立target scorer，因为本阶段仅冻结源诊断；若后续训练，按新run预登记并自动七环境评分。

本地4项测试（含9个checkpoint负例、3个对齐负例）PASS。唯一独立P0/P1审查PASS，确认原生头、source V对齐、既有IQ复用、无训练/query路径。REJECTED_EXTRA_GATE：额外设计审批、文档/成员哈希、重复审查不阻断。原始预测、日志、权重和私有连接信息不进Git。


## 技术失败，原产物保留

r01 已 FAILED：两份真实 checkpoint smoke 均 PASS；尚未生成外部预测。Torch2.1/NumPy2 的 C 数组桥产生了不同的 ndarray 类型，array_equal 报 TypeError。小型无 IQ 脚本在原环境的原生 forward 前后均复现；tolist/frombuffer 路由通过。无训练更新、无 query，不作为性能结果。代码 9309a4988。修复后使用新 r02，不覆盖原目录。
