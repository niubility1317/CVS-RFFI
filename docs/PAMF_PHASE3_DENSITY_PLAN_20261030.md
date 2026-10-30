# PAMF 源侧多中心密度候选

固定 run：`20261030-phase3-star-density-s2026092701-r01`。本轮检验冻结原身份的已知多中心概率支持域及辅助背景密度比。唯一主为 `density_ratio`，`known_only` 为匹配无 OE 控制。此前联合邻域结果只说明机制值得检验，不用于挑容量、分数或阈值；不复跑旧 8NN 距离比或以最近中心距离商冒充新模型。

原完整 Tri-SIG-TIME SSL F0 E50/11100/default6head/232993 不更新。仅复用原 F0 在既有 G10MIX01/P1 上的特征与默认输出：protected-open code7dd90671968ddf58aab5a5b302b4d56371c8b9c9 的 L6300/OE6134 特征；已冻结 local-support code84cdc012f7d9641b283c407f6048bfe4e6e21cf6 的 V27000/novel3150 `unit_features`、默认 logits 和公共元数据。旧拒识分数、旧阈值和评分反馈不作为输入；local-support 是原 F0 特征复用来源，不是训练父。

L 为旧六设备各1050，每设备×五源RX×三日期70条。OE 为21个辅助设备的6134条，仅 outside-six 成员角色及公共 RX/day 可见，真实设备标签/K隐藏；算法32个背景mode容量不是未知TX数。V旧六类各4500，仅冻结分布后的既定已知风险校准；source novel19-15/19-16/7-1各1050，仅独立truth-last评分。原Us56700只属于Phase1祖先、本轮不读。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。源RX/日期已见、novel历史已曝光，不是新RX/TX盲测。目标U/Q、正式RX24-18、clean/P0 IQ新读取0。

全部物理观测沿原固定G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/no additional EQ、referenceSNR30dB/reflect；郊区城区各0.5及六分量面积概率，receiver2027/scene2026103001/channel2026104001，原始信道前一次unitRMS、输出不逐样本RMS。这里只在缓存特征上计算，不生成IQ、不运行G。输入特征在fit与score均以同一FP64函数单位化，禁止不同精度路径偷偷改变模型定义。

## 事前固定的统计模型

维度D=160。仅在L上计算全局均值o以及六类×五RX×三day共90格的均值。每格70条，不能各拟合独立全协方差。令C为所有格内去均值残差的population pooled covariance，`C_s=.9*C+.1*max(trace(C)/D,1e-8)*I`；`W=inv(cholesky(C_s)).T`，统一空间`z=(unit(x)-o)@W`。共同白化、已知mode与收缩基准完全不接触OE/V/novel/target。

已知90个mode使用格内z均值、权重1/90。定义`v_pool=mean(||z-mu_cell||^2)/D`，各mode径向方差为`max(1e-8, alpha*v_cell+(1-alpha)*v_pool)`，`alpha=70/(70+D)`。相同已知state供两policy使用；known_only不包含背景参数，OE不影响其拟合或得分。

仅主模型读取OE特征，在同白化空间使用公共15个RX/day域总质量各1/15的样本权重。固定weighted kmeans++、seed2026103019、capacity32、25次Lloyd，不早停、不按novel或V选容量/轮次。空mode保持中心，最终零质量mode不部署；kmeans++剩余距离总质量为0时按稳定行序选择未选中心，只处理算术退化，不输入真实TX/K。实际非空mode数仅为统计分量数。

背景mode权重为域平衡的分量总质量；其`v_mode`为加权径向残差均值除D；`n_eff=(sum w)^2/sum(w^2)`、`alpha=n_eff/(n_eff+D)`，同样向仅L的`v_pool`收缩、floor1e-8。不使用OE TX均衡、伪身份或未知设备配额。保存完整固定25步目标/空mode/中心变化轨迹，以实际执行为准。

对任一混合分布，计算 `log p(x)=logsumexp_m(log pi_m - D/2*log(2*pi*v_m) - ||z-mu_m||^2/(2*v_m)) + log|det W|`。mode概率质量、体积、尺度均实际进入似然。control=`log p_L`；主=`log p_L-log p_OE`，数值越大越应接收。没有另训六类身份头，身份始终为原F0同P1 defaulthead argmax。

只导出o/W、mode均值/方差/权重等聚合数组及六阈值；不部署source逐行ID/特征/IQ。小规模源反证使用缓存，并不证明未来实际G+F的时延/显存或已知保护。

## 自动预测、评分及资源

launcher首步实际原F0 checkpoint两条合成G10无query smoke；参数/buffers不变、G不加载。PASS立即继续cached L拟合；control先冻结，主读取OE并冻结。新真实F前向0、合成F2、G0，V27000/novel3150特征逐值复用；原模型和缓存构建成本不称免费。分布拟合是CPU统计/25步聚类，不冒称2220神经更新或新训练epoch。

两模型冻结后仅在known V按既定`star_class_risk.calibrate_known`执行六predicted-class/common-tail阈值，保护原同P1身份的42个既定组、损失≤2pp及FRR≤10%。两policy全部阈值冻结后才消费novel特征、完成两套预测；不读取private truth。source V不更新分布/白化/中心/方差/权重、不早停。

独立CUDAempty/BLAS1 CPU scorer在全部预测完整后唯一连接novel truth，报告宏A_K/C_K/R_U/H、AUROC/OSCR、原F0损失、逐真TX/RX/day/scene及配对。主density_ratio同时通过known风险与source R_U≥90%才另预登记跨未见RX星地开发评估；控制不能替换主。source失败不停止合法评分或按分数重做，不降低目标、不把source通过当跨RX成功。

v3主目标仍是跨RX星地C_K≥90%、R_U≥90%、A_K距同输入原F0闭集≤2pp，主报宏A_K/R_U/H；后续无标签适应、未知发现、未来A_N≥80%与注册旧损≤2pp保持ACTIVE。目标卡旧95/5不改写最新用户明确的v3指标；95%仍可附报研究目标，不能假装已同步该工具字段。

一个GPU smoke/prediction worker实际启动free≥4096MiB，每卡本任务新增≤2；其后CPU统计和独立评分。其他用户进程不停止/修改/重启。一次代码commit、聚焦负测及唯一直接P0/P1审查、最小登记、单N607preflight/archiveSHA/compile，启动后独立PID/CWD/argv/GPU/log及终态读回。非法来源/角色、非有限、碰撞或技术失败保留，不自动重试；UNKNOWN先核原PID/root。预期原始日志、模型聚合NPZ、阈值、两policy预测和独立score产物；不增加hash/receipt/签章或报告批准门槛。
