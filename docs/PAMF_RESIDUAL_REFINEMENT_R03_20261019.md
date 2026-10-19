# 冻结粗恢复后的恢复残差修正

状态：RUNNING / VERIFIED（源信道缓存生成）；11项新增与12项R02回归检查、compileall、controller --help通过，独立P0/P1无阻断。尚无本轮训练或有效性结果。承接用户“六类恢复器”报告 R6，并保留确定性二次修正控制。有限 L 身份保护和 F 联合训练本轮不启用，优先保持恢复训练的无 TX 标签边界。

父恢复器固定为 R02 源波形规则 A 选中的 `tf_unet_mixed`，不是根据目标分数选择。复用 R02 的源角色、原 F 来源和仅源选模记录；父模型从零 E20、6300 L_s/56700 U_s、0 TX 标签训练、无目标拟合。R03 只读取其 source_decision、最终 checkpoint 及源数据，禁止把父目标指标输入训练、尺度估计或选模。

三行是 `residual_deterministic`、`residual_diffusion`、`residual_flow`。共同冻结 b=R_base(r)，训练 e=(x-b)/s，其中 s 仅由 R03 第一轮的固定8192条 L/U 观测估计一次 RMS，作为 checkpoint 常量。模型最终输出 b+s*e_hat；e 是恢复残差，不称为真实硬件指纹。推理仅用 r、b 和预登记噪声，不读取 x、真实信道、场景、TX、RX 或真实残差幅度。

确定性修正以(r,b)为四通道条件、零输出投影开始。残差扩散使用64级cosine/v目标、16步DDIM；残差流从0.02*noise的归一化残差出发、16步Euler。生成式条件在每个尺度注入，每次采样编码一次；不复用完整IQ生成的起点。无潜空间压缩、无输出硬剪裁、无F损失、CE或PL。

数据为全部63000源物理样本，每轮一个观测，15%clean、60%residual/post_sync/noEQ、25%full/post_sync/noEQ。full/residual从同一个原IQ分别生成并共享latent；R03使用新的namespace，每轮新信道，不复用父训练噪声。源V两路固定面板直接复用，不重验证数据。每行scratch修正器E20/4940更新、1260000次曝光；父恢复器E20成本及原F历史成本分别列出。另有一次8192条共享尺度估计、零优化更新。

优化设置沿R02：AdamW、LR2e-4、247更新warmup、cosine到2e-5、weight decay1e-4、clip1、FP32。确定性损失IQ NMSE+0.1稳定化log-power；生成式velocity MSE+0.2同一恢复损失。三行同数据/噪声流/预算，不因结果调参或停训。

源规则A仅比较粗恢复器与三修正器：clean NMSE<=0.02后最小六residual平均原坐标NMSE，固定顺序打破并列；不读取F/TX字段。规则B单独披露为有标签source V辅助选择：相对冻结粗恢复器low urban至少+0.5个百分点、clean下降<=0.2个百分点、六场景平均NMSE不恶化、clean NMSE<=0.02；生成式128条源样本三次采样的任意类别切换比例另须<=5%。B无合格修正时不替代为目标选择。最多确认A/B去重后的两个新修正器；A选粗恢复时复用父预测，不重复生成。只有纯源资格成立才允许后续多seed/完整正式扩展。

源选择冻结后自动在父R01原1200条固定输入上完成clean+六practical预测，再独立truth-last评分；仅作为已有暴露基准上的阶段性配对诊断，不冒充完整正式received IQ或在轨验证。无资格且A仍为粗恢复时记录源证伪结果，不强制访问未选候选目标。低分不触发技术停止。每行完成源双路评分、完整采样条件置换、batch1延迟及采样稳定性。

N607计划GPU0/1/5各最多一条本run worker，总并发三，其他owner计入cap2/free12GB。16CPU生成新流；独占release/output，先共享尺度估计再训练。实际路径、commit、命令与停止规则随最小experiment.json登记；不增加额外审批、hash或重复review。

|ID|来源|要求|目标文件|状态|验证/边界|
|---|---|---|---|---|---|
|R03-01|R6.1/3|固定合法粗恢复器、无梯度/状态更新|models/runner/controller|verified|冻结参数/BN状态负测与父来源负测通过；实际父source决策/commit已读回|
|R03-02|R6.3|全候选共享源训练残差尺度|runner.calibrate|implemented|固定8192索引及尺度/偏移数学测试通过；实际源尺度待GPU生成|
|R03-03|R6.4|确定性二次修正对照|models|verified|初始等于粗恢复、优化器排除父参数、输出不硬剪裁|
|R03-04|R4/R6|残差v目标与完整DDIM一致|models|verified|残差v目标/重建尺度正确；完整采样条件编码一次，无状态更新|
|R03-05|R5/R6|残差流训练/起点/积分一致|models|verified|归一化残差起点及训练/推理一致性通过；不是I2SB|
|R03-06|数据5/训练6|新namespace共享full/residual、全L/U|data/runner|implemented|原生新namespace不同且重放一致；20轮全量生成待启动|
|R03-07|统一接口|单IQ派生原PAMF全部分支|原predict_row+新loader|implemented|可移植复合checkpoint及原predict_batch接线通过；实际F读出待启动|
|R03-08|验收|源A/B与父控制、资格后确认|controller|verified|A拒TX/F/NaN、相对父控制资格及生成稳定阈值测试通过|
|R03-09|生成验收|三采样类别稳定性、编码置换|runner|implemented|完整采样及三draw接线；实际训练后诊断待运行|
|R03-10|资源/日志|损失/曝光/实际延迟/显存|runner|implemented|训练/分层日志及同步batch1接口已接线；实际成本待测|
|R03-11|最小流程|负测与真实源checkpoint smoke|tests/controller/runner|implemented|23项本地检查通过；launcher先真实F源smoke，首训练更新真实R往返|
|R03-12|最小流程|单次P0/P1、commit/预登记/发布|experiment记录|verified|单次review/提交推送/归档比较/compile/真实F源smoke与live PID/CWD/log增长通过|
|R03-13|最终证明|完整正式输入、独立族、多seed|后续冻结验证|deferred|本run单seed小面板不能替代|
|R03-14|R3/身份保护|有效算子展开/有限L或联合F|本轮未启用|deferred|独立权限与控制，不混入纯自监督|
