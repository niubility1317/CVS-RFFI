# 可信纠错固定规则：三种子源域可靠性诊断

本次先检验“教师可信、学生尚未学会”的新样本能否提供可靠监督。固定规则若未达到源域可靠性与样本量标准，就保留负结果，不启用 B/D 纠错训练，不从 V 或目标结果搜索阈值。完整矩阵、命令、路径及逐行配置见 [experiment.json](experiment.json)。

## 固定机制与诊断边界

保持 PAIR 原置信度、物理邻近门控和学生强视图一致性接受掩码。教师可靠性另用两个相关弱视图的同类预测且两者最大概率均不低于 0.99 判断；新增微弱高斯视图噪声 RMS 为包 RMS 的 0.003，使用私有随机数生成器。高概率和相关视图稳定不是校准正确率，也不是两份独立证据。

新纠错候选是原 base gate 合法、教师可靠、学生强视图 argmax 不一致的包。逐批按教师概率排序，最多接纳原接受数的 10%（向下取整），新包原强视图硬伪标签 CE 总和不超过原接受包 CE 总和的 25%。没有原接受包时不补充纠错，无强行补满或类别配额。全部接纳包保持原均匀硬 CE 及 accepted-mean 归一化；不改 λ_U=0.16，不加入软目标、梯度重加权、提前 U 或支路弱化。

读取已冻结 PAIR 固定 final50 student 和 ownEMA，严格加载原结构。27000 个完整 V 包在每个模型 seed 各观察一次，共 81000 个模型-包观察、243000 次包前向；同一物理 V 集被三模型复用，不冒充 81000 个独立样本。教师 eval，学生 train/no_grad 复现原强噪声及 Dropout；检查参数、buffer 和 .grad 无变化。源标签在全部掩码固定后才用于计数与分层。U IQ/真值、目标 IQ/真值均不读取；不拟合模型、分类头或阈值。

## 矩阵与预定判据

| 行 | model seed | PAIR 父模型 | 本次更新 |
|---|---:|---|---:|
| TRUST_PROXY-s2026092701 | 2026092701 | 同 seed fixedfinal50 student/EMA | 0 |
| TRUST_PROXY-s2026092702 | 2026092702 | 同 seed fixedfinal50 student/EMA | 0 |
| TRUST_PROXY-s2026092703 | 2026092703 | 同 seed fixedfinal50 student/EMA | 0 |

在任何本次源代理结果读出前固定：合并新接纳包数至少 100，精度至少 99%；每个 seed 至少 20 包，精度至少 98%。这是可否启用 B/D 的源域代理判据，不能证明较早训练时刻或真实隐藏 U 的精度。记录 original、可靠且已一致、纠错 candidate/admitted 的覆盖及正确数，按 RX/day/TX 分层，保留原置信度、物理邻近、强一致性、新教师置信度、新弱视图置信度、新弱视图一致性的全部 64 种共同拒绝组合。保存源级决策后独立重算完整统计。

CE 与逐包 logits 梯度范数只用于判断学习信号，明确区别于在线 fθ 梯度和实际参数更新。后续训练必须另记实际 U/L 骨干梯度、夹角、clipping 及参数变化，不以本诊断替代。

## 下一步训练的独立问题

若固定源判据通过，第一项训练采用 PAIR 2×2：A 保留强一致性硬否决/保留全局 U entropy；B 可信有界纠错/保留 entropy；C 保留硬否决/关闭 entropy；D 可信有界纠错/关闭 entropy。默认联合分类头、50 epoch 的 33L+17U 安排和均匀硬伪标签 CE 保持；新增弱视图计算与随机数安排在四行匹配。能力触发、CVS 证据支路教学、跨 RX 更新效用均另行消融，不在此混合。

## 发布与技术状态

唯一 launch owner：`codex/root/trusted-correction-source-20261014`。普通 N607 账户，CPU 运行、2 线程，禁止占用 GPU 或操作其他健康任务。运行根 `runs/20261014-diagnostic-trusted-correction-source-manysig-m3-r02`，发布根 `releases/cvs_trusted_correction_source_20261014_r02`。执行命令为 `python -u -m experiments.cvs_trusted_correction.audit`，环境和绝对路径见配置。代码身份为发布归档中的 Git `release_commit.txt`。

已完成固定策略负测试、accepted-mean 复现及私有弱视图 RNG 检查；一次独立 P0/P1 审查 PASS。launcher 首步实测真实 PAIR student/EMA strict-load/no-query smoke，PASS 后立即诊断。错契约、输出冲突、非有限值、原 gate 不一致、参数状态变化或保存统计重算不一致即停止本次诊断并保留产物，不自动重启。

当前状态：LOCAL_VERIFIED，尚未把预定诊断写成已得结论。发布及结果以独立回读证据更新。


## R02 技术修复与新主线

仅将保存完整源决策由 mixed NumPy NPZ 改为 native tensor source_decisions.pt，加载后重算全部记录。已实测最小失败与修复后的相同 tensor 往返，一次定向 P0/P1 复核 PASS；不变代码不重审。R01 保留，R02 独立输出。最新用户明确主攻“跨 RX 关系自蒸馏＋增量证据响应”，本诊断仅收尾，后续不自动继续完善伪标签接受链。关联来源是实验方案而非已验证效果；仍不宣称达到84%。


## 完整源结果

独立回读 VERIFIED/ANALYZED。seed01/02/03 candidate39/34/25，新admitted15/15/11，全部41/41正确；原accepted24414/24057/24413，correct24411/24057/24413。源决策完整保存并重算，3×27000模型观察、243000前向、27000唯一物理V复用。固定pool>=100、each>=20条件不通过，故source criterion=false；不是技术失败，也不证明隐藏U精度。无U/target读取、无拟合与阈值调节。详见result_evidence.json及remote各row analysis/source_decisions.pt。新主线继续关系学习，不自动启动gateB/D或搜索阈值。
