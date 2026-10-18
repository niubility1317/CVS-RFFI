已完成：SOURCE_ANALYZED。实际源域状态结果、分层图、负结果与限制见[完整分析](final_analysis.md)。以下保留原预登记历史，不将其PLANNED文字作为当前状态。

# E30→E50 源域物理状态分层补充

状态：PLANNED，未发布／启动。run=20261018-diagnostic-FcrossRX-source-state-budget-e50-s2026092701-m4-r01；唯一owner=codex/root/source-state56。一个CPU分析任务覆盖4组已经保存的NOAS30／AS30／NOAS50／AS50默认头源预测，model seed2026092701。

目的：补齐已有预算收益的源域状态与质量落点，不新增机制、不选模或晋级。E50单seed正式成绩已经暴露，本分析不声称预注册的盲态机制验证，不读取formal文件，也不据此改变正在健康运行的E50四行。

输入：P43原E30与P50原E50各自source_qualification/predictions.npz，及固定source panel的data_roles.json、observations/completion.json、physics.npz和physical_records.jsonl.gz。源V1800、原6300／56700隐藏／27000角色、全部5源RX与3day、split392005不变。两draw合并仍只有1800个独立物理样本和一个模型seed。仅使用condition0=full，严格核对ids／TX／RX／day／draw／scene／条件顺序；不读取IQ、checkpoint、U真值或正式truth/predictions。

固定比较为As50−As30、NoAs50−NoAs30、As50−NoAs50、As30−NoAs30及预算交互。固定clean正确集合是NOAS30源panel，不使用连续E30的集合。每个draw及合并状态／锁定／接收质量／AGC／边界层均保留；稀疏与空cell不隐藏。噪声分箱沿既有P52 <3、3至8、8至15、≥15dB（signal_to_added_noise_db，非总SNR）；其余少量固定箱在本次代码提交中披露，不根据分类结果拟合。

默认头仍是ResidualPhysicalHead，scale=30.0普通float不可学习，训练weight保持原联合方式。本分析同时报告原始margin和除30的cosine margin，不新增头或训练。状态关联不是损伤移除的因果效应，不将P53的状态CM冒充本E50结果。

执行预算：0新model-epoch／更新／NN／IQ／CP／模型预测／正式评分。只以NumPy重分层已有源预测，CPU2、CUDA隐藏，Matplotlib绘图；墙钟实测。父训练与原预测成本保留原报告，不再次计作本分析新增成本。

命令：CVS-RFFI/python -u analysis/cvs_budget50_source_states56_20261018.py --project PROJECT --out PROJECT/runs/20261018-diagnostic-FcrossRX-source-state-budget-e50-s2026092701-m4-r01/result。CWD=PROJECT/releases/cvs_FcrossRX_source_state_budget_E50_20261018_r01；原始输出根PROJECT/runs/20261018-diagnostic-FcrossRX-source-state-budget-e50-s2026092701-m4-r01不可覆盖。实际代码commit、最终图名、一次有限检查／独立P0/P1、发布与PID证据在启动前补齐。

此阶段没有模型／checkpoint／IQ消费，因此真实checkpoint/noquery前向smoke不适用；使用聚焦fixture确认源ID/条件对齐、配对数、cohort和分箱边界。一次普通账户预检、一个release archive SHA本地／远端比较、远端compile及启动后只读PID/CWD/argv或终态产物核验；不新增权限token、逐文件hash或额外审查。

停止规则：源角色／样本／draw／条件错配、非有限logits、shape不符或技术失败时保留现场并停止本CPU分析；无科学效果阈值、无自动重跑、不影响任何健康训练。输出CSV／JSON／CM、mid/low状态与质量PNG/PDF、命令和真实wall及未完成项。

本次固定分箱：质量／新增噪声[-inf,3,8,15,inf]dB；waveform−white[-inf,-3,-1,1,3,inf]dB；绝对残余频率[0,100,1000,10000,inf]Hz；噪声着色[-inf,1e-9,3,10,inf]dB；残余核能量[-inf,-20,-10,0,inf]dB；interior_count[0,128,192,224,256,inf]；AGC二值。均在实际重分层之前固定；不根据分类结果调整。最终图名为practical_mid_urban_source_states与practical_low_urban_source_states的PNG/PDF。

实际分析代码提交：a5d6ba378b5f20b3d443126cb6e8d0b3793b1bc3；一次独立P0/P1 PASS，10项限定fixture及compile PASS。一次N607预检VERIFIED（普通账户、新run/release目录、四组既有source logits及纯metadata、NumPy/Matplotlib）。源域实际结果尚未读取。
