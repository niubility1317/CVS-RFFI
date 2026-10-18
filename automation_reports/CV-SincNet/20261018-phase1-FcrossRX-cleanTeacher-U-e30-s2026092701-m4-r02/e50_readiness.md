# P65 总 E50 续训准备（只读核查，尚未晋级或启动）

用户已授权对源域证据支持的好方案使用总 E50。核查时 E30 四行健康运行；本记录不改变其科学条件，不依据正式成绩晋级，也不是新增门槛。只读核查 0 NN／IQ／checkpoint／更新／远端操作。

可复用 `experiments/cvs_practical_budget50/source.py:15–33`（cvs_iq_timing_20261016 worktree）及本 worktree `experiments/cvs_satellite_budget62/source.py:15–33`，最终调用 `experiments/cvs_crossview_supcon41/source.py:63–163`。语义为严格加载本行 FINAL30 学生和自身 EMA，再重置 AdamW、cosine20、时序缓存；新增20轮/4440更新，累计50轮/11100更新，不恢复原optimizer/scaler/RNG轨迹。pure_game专用 xuc_fusion/resume 不适用。

若源判据通过，后续独立 adapter 必须处理：

- 学生／自身 EMA 来自本行 P65 r02 FINAL30；两冻结教师仍是原 P43 B01/B00 FINAL20 的 ema_model，不能把新的E30父权重误当As20教师。当前 observer.initialize 使用self.parent作As20 payload，需要拆开。外部bank保持，不进部署模型。
- 新段local1–20/累计E31–50/全局step6661–11100；同步 design 更新数、model step上界及+4440/+21、channel E21–30窗口、observer/protocol终态和分析成本。训练随机流刷新并和已用段分离。
- 教学年龄应为11–30，成熟权重保持.16；ZERO始终0，不能重启五轮ramp。L/rawslot/SupCon使用累计偏移30/6660。ordinary增强继续引用原scratch20参数和native E13冻结日程，不把新父checkpoint epochs10当原训练日程。
- 分开核查来源：E30父模型是本行r02/local epoch10/EMA2220/final_only/206keys；两个教师是P43 scratch20/EMA4440/B01或B00/无初始化继承。继续核对同一物理角色和完整合法祖先，不解封U真值。
- 新20轮预计额外U学生568240NN、冻结教师合计1136480NN，实际以计数为准。学生总E50，额外复用NoAs20教师后单行独特系统前置成本70模型轮次；共享历史只计一次。配套同seed/同E50无As对照必须核实来源/日程/角色，不能把不同前向成本全部当机制收益。

另已核对教师复制与成本实现：P43 Observer.initialize无encoder替换/闭包；full计数通过绑定MethodType写各clone自身整数，forward_identity_only调用self.id_backbone。临时特征hooks在finally移除。冻结bank手动计量；原EMA独立捕获counter offset，P65最终仅减一次extra学生再加frozen计数。未发现共享原student前向或双计数问题；未为此重复跑NN。

本次没有实现E50、增加seed或启动实验。最终矩阵在源结果返回后按必要匹配对照固定。

最终状态：r02全部四行E30及七环境评分已完成。主候选源域判定FAIL（clean超0.1pp损失且未优于同龄冻结As教师），本次路线关闭，未实现或启动E50、未扩种子。本页只保留接口复用知识，不是待执行实验。
