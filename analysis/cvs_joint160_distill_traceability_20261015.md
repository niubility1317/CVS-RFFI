# CVS clean→urban 身份方向蒸馏设计追踪

来源：用户 2026-10-15 附件 `E:/codex/home/attachments/17f04499-11a5-4db8-b172-65bd1ebd75e5/已粘贴的文本.txt`。当前 E50 四行及所有历史产物保持。新增首轮复用 C3-s2026092701 为 A，B/C 各从头 E50、相同 model seed、33＋17、11100 更新，零父训练继承。只比较一种额外约束：A 单侧分类间隔、B 默认头 W 停止梯度的输出 KL、C 实际 joint160 的单向方向损失。B/C 替换 A 的间隔约束，不与其叠加。原 paired CE、旧门控半监督和网络/头保留。

本记录用于落实用户设计，不增加审批、签名、hash 或实验门槛。代码与当前健康训练 release 分离；新参数只由合法源 IQ 与实际梯度校准，目标分数不进入配置。首轮不增加 U urban 训练前向，以免将一致性对象与额外 U 监督量混成一个因素；原 U 训练保持。后续 U feature 约束单独列出。

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| J01 | 三.1 | 直接作用于实际部署 joint160，非旁路或新 MLP | experiments/cvs_urban_identity50/features.py | verified | 原/辅助特征及默认头输入一致 | 已核实实际 key=id_feat_joint |
| J02 | 三.1/3 | own EMA clean 目标 stop-gradient，L CE 全程、错误 clean 教师不作强目标 | observer.py/objective.py | verified | 教师无梯度；低置信/错误 urban 学生仍有辅助梯度 | L 只使用教师 clean 正确这一条资格 |
| J03 | 三.2 | 只约束归一化方向，eps 与原 cosine 头一致，不匹配原范数 | objective.py | verified | 缩放不变、零向量有限 | cosine eps=1e-4，scale30不变 |
| J04 | 七.1 | 辅助损失对 W 梯度为零，原 CE 继续训练 W | objective.py/checks.py | verified | 实际默认头 aux 无梯度、CE 非零 | B/C 同一 W 边界 |
| J05 | 七.2 | 辅助 gain 停止梯度，branch 本身不得 detach | features.py/checks.py | verified | forward 相同；aux gain0；t1/t2/PA/ref/SIG 梯度非零 | 仅改变辅助梯度，不改部署/CE |
| J06 | 七.4 | 合法源校准五组梯度比，目标中位数0.1，冻结系数，不在线均衡 | calibrate.py | pending | 无 optimizer/状态变化；source-only 数值记录 | source frozen C3/own EMA，只读 |
| J07 | 七.3 | 主 Dropout 保持；诊断 eval；清楚区分信道与随机删除 | observer.py/calibrate.py | pending | train/eval 前向等值、模式恢复 | 不移动 t2 读点 |
| J08 | 八第一轮 | A/B/C 除目标外匹配网络、数据、EMA、预算与增强 | design.py/observer.py | pending | 原结构 state_dict 相同；同源固定 eta 日程和命名 RNG | A 的 source 日程复用于 B/C，不读取目标分数 |
| J09 | 八/验收 | 自动逐行 clean＋六 practical、独立 truth-last、默认头、无目标反馈 | dispatch.py/evaluate.py | pending | protocol-negative+真实checkpoint smoke；真实进程/产物 | 正式 IQ 不生成/调弱 |
| J10 | 七/八 | 实际骨干梯度、更新、裁剪与覆盖、完整成本 | observer.py/report.py | pending | 全量日志及 actual 参数更新 | 预算耗尽不代表收敛 |
| J11 | 八验收 | 正确/错误 TX 特征距离、固定 clean 正确集合修复/新错、分支方差与 gain | mechanism.py | pending | 无拟合参数的同物理样本分析 | 单 seed 不声称复现 |
| J12 | 二/六 | 不匹配 raw29、原始PA统计、所有分支或所有gain | features.py/checks.py | verified | 损失路径实际边界检查 | 固定响应允许真实信道变化 |
| J13 | 三.4 | 有容差方向目标作为独立后续比较 | 后续独立候选 | deferred | 第一轮不启用 | 用户要求先验证简单方向损失 |
| J14 | 四/八第二轮 | 同信道高质量中间视图：H/几何/RX/接收状态/补偿/边界/延迟/裁剪固定，只在原注入处改变新增噪声 | 后续独立候选 | deferred | 需源证据后与 D/E 匹配前向 | 不修改正式测试，不重新采全信道 |
| J15 | 五/八第二轮 | SIG_TIME 增量6、面积15、特征时间面积6分组；常数dt排除；源尺度下界；梯度进入local_projection/t2 | 后续独立候选 | deferred | 先验证 joint160，不能直接错位对齐112维 | 描述子和维度保留 |
| J16 | 六 | PA/ref 投影到同信道中间视图的局部约束 | 后续独立候选 | deferred | 仅源证据支持后独立加入 | 第一轮只经实际joint160学习 |
| J17 | 三.1 U_s | U 的特征目标复用 clean 教师资格，不加学生 urban 门控 | 后续独立候选 | deferred | 首轮原U旧门控保持，不新增urban U视图 | 避免改变 A/B/C 监督数据量 |

初始审计：现有 paired_margin 是类别间隔约束，未直接对 joint160 蒸馏。实际融合为归一化主路径＋tanh(PA gain)×PA＋tanh(response gain)×response_projection＋0.2tanh(SIG gain)×SIG投影。可在实际 satellite 前向捕获三个 gain 前的特征，以零值梯度修正保持数值不变，只取消辅助 gain 梯度；不得 detach 三个特征。原模型和默认头参数结构保持。

训练发布仍使用既有最小流程：合法数据/来源、代码配置 commit、定点负例与真实 checkpoint smoke、一次 P0/P1、最小登记、不可变输出与传输/编译、启动后回读、独立评分。此文件本身不是额外发布门槛。

本地证据：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.checks`，7项PASS；旧 `experiments.cvs_urban_sigtime50.checks` 10项PASS。原始 stderr 位于 local_artifacts/cvs_urban_sigtime_e50_20261015/identity_checks_stderr.txt 与 identity_native_compat_stderr.txt。J02这里只证明损失/资格边界，实际 native 训练接线仍由 J09/J10 验证，不能以unit结果冒充真实训练。
