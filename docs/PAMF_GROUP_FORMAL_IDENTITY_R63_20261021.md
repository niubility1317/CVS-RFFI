# R63：整组压缩的固定无 F 同 TX 关联

状态：LOCAL_VERIFIED，尚未发布。30 项针对性检查、CLI 和一次独立 P0/P1 审查通过。R62 两行仍在固定源训练，本设计未读取其源验证或正式准确率，也不改变模型、损失、超参、预算或测试次序。

本轮只检验已冻结恢复 IQ 的同 TX 关联，不新增恢复网络。读出固定为 raw、指定 M35、同增量预算 R49 retain_refine、R62 GN3/GN1 两行。R62 两行均完成七景正式预测和独立评分后，才读取其已保存 IQ。raw/M35 的 56 个 R56 解析单元只读复用；R49 与两 R62 共 84 单元新算，总 140 单元全部固定后独立接入 TX/RX/day 真值。

原 R34 的 LU8192 无标签源统计保持，不重新拟合任何坐标、阈值或描述子。镜像 64、谱 4、参考 24、组合 92 维解析特征、std floor .05、每 TX/RX/day 取最低两个 opaque physical IDs、参考与 query 同时跨 RX 和 day、六类完整单元等权均沿用。199 次 RX/day 内 query 标签条件置换的均值/95% 界是描述性对照，不称 p 值。保留每 TX、RX、day 和全部正负差异，特别保留原 clean 参考弱 TX2。

正式数据为原 168000 条物理样本、clean 与六种 practical，郊区和 urban 的 high/mid/low 全覆盖。预期 2016 个分组记录、336 条件对照、140 配对差；主配对为 GN3/GN1，两行各与 M35、R49 比较。R49 与新行同父、同 E20/4940/globalmicro256/IQ 教师，但宽度和归一化不同；不把它称同图。新算解析输入 3528000，无新神经预测、F 权重/输出/特征、checkpoint、训练或分类头拟合；新校准为 0。

N607 普通账户、CPU2、空 CUDA、GPU worker 0，沿用 R41 score.lock 最多一个本任务 scorer。独立不可覆盖 root，单 Git method commit、一次真实两条 L_s 无参数 no-query smoke、一次独立 P0/P1 审查，最小登记/单归档 SHA/compile/实际 PID/CWD/argv/日志核验。启动入口 python -u -m experiments.pamf_group_formal_identity.runner run，准确参数与路径登记在 experiment.json。预期保存 resolved、smoke、features、physical_ids、calibration_reference、feature_complete、summary、complete 与日志。父失败、输入/ID/预算/分组不符、非有限值、输出冲突均保留产物，不自动重试，不改健康任务。

该检验不阻塞 R62 正式测试或独立权重交付。整体关联、条件置换差异、识别率和 F 几何都不足以证明每个恢复细节真实属于输入 TX；不外推真实在轨场景，不称新独立盲测，不用于回流选模或选择性重跑。原性能与逐细节真实性目标 ACTIVE。
