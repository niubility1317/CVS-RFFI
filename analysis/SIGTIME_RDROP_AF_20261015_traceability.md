# SIG_TIME＋RDrop A–F 实现与实验追踪

来源：用户 2026-10-15 最新消息。此记录优先于此前 C 单项准备；全部实验使用默认分类头联合训练。不得把标量公式检查作为网络结果。

| ID | 来源 | 验收要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| R01 | §1 | 居中坐标 p=p0+25000q，Fs=25000000，q初值0；复用既有模块 | coordinate.py/model.py | verified | 六实际网络构造与初态物理状态匹配 | E/F |
| R02 | §1 | lr_q=alpha Fs/s，eps_q=epsilon s/Fs，跟随 cosine，不叠乘历史倍率 | optimizer.py | verified | CPU/CUDA 实际 step，3 origin×100步 float64误差2.2352e−8 Hz | E/F |
| R03 | §1 | 全局 clip5 前恢复 g_p=g_q/s，裁剪后恢复坐标梯度 | optimizer.py/observer.py | verified | 全局物理norm相同，CUDA普通参数首步最大差3.5763e−7 | 含所有普通参数 |
| R04 | §1 | 物理衰减含 origin，无双重衰减 | optimizer.py | verified | 含origin实际更新等价，无origin反例8.32/332.65/6320.26 Hz | E/F |
| R05 | §1 | student/EMA/推理/导出统一坐标，真实叶版本缓存失效 | coordinate.py/model.py/feature_check.py | verified | 五个真实final50 student/EMA/checkpoint核与joint160导出/去W检查通过 | 默认头不变 |
| R06 | §2 | A–F 固定矩阵，A/C 仅核实匹配后复用，健康任务不重训 | design.py/provenance.py/dispatch.py | verified | 18行登记；A三实际配方/祖先核实且CP smoke PASS；C无匹配旧任务，BCDEF首种子健康启动 | C 当前未启动 |
| R07 | §2 | B/F 复用 SIG_TIME_X3 独立三路，共同 joint160，无统计分支/投票 | model.py | verified | CPU/CUDA三独立投影非零梯度，实际joint160，复用原CompositePool | 禁止替换单路基座 |
| R08 | §3 | RAW_HZ/普通增强/LS.01/RDrop1/旧门控 ownEMA cos，同次增强IQ两前向 | design.py/checks.py | verified | CPU/CUDA及实际source_mechanism复用增强IQ，A三seed远端实际配方一致 | 默认头联合训练 |
| R09 | §3 | scratch50=33L+17U，11100步，三seed，U真值隐藏，数据角色不变 | design.py/provenance.py/source.py | implemented | 固定配置/负例通过，实际运行待返回 | 不用训练 checkpoint 初始化 |
| R10 | §3 | 必要首种子矩阵完成后，读取新target汇总前冻结并上传源域扩种子决定 | gate.py/dispatch.py | verified | 源决定4173ae34已push/远端OID核实后放行十行，再回收首n1新target；两条顺序8例通过 | 参数不根据target修改 |
| R11 | §3 | 逐行冻结后自动 clean 预测，独立 truth-last 评分 | evaluate.py/dispatch.py | implemented | 逐行自动预测与独立CPU评分接通，待运行 | 每行单owner |
| R12 | §3 | 配对 B−A/E−C/E−D/F−E/F−B 与交互，均值样本SD/正差数/F1/最差RX/day/成本 | evaluate.py/report | implemented | 六比较/交互/SD/F1/worstRX/day/成本已接通，待完整结果 | 负结果保留 |
| R13 | §3 | 实现配置结果 Git提交推送并核对远端OID | prereg/report/catalog | implemented | 09554c3a执行代码已推送/远端OID核对；首轮实际证据在本次结果提交，完整矩阵结果待回收 | 私有产物不入Git |
| R14 | §3 | 最终实际回答 B/E/F 有效性，D差异不单归因LR | final report | pending | 待完整矩阵 | 不在C完成时收尾 |

实现严格遵循用户公式和矩阵；任何不可达项会记录具体证据，不能用近似替代而不披露。
