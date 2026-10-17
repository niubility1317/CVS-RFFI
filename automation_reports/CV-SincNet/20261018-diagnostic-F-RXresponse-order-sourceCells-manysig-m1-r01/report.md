> 已 ANALYZED／VERIFIED：逐包读出 SSE2.945018，旧均值2.944634；0/18折、0/5RX优于加性控制。归一化顺序解释失败，固定方向关闭，0拟合/NN/更新/target，非联合目标完成。见 [完整源分析](source_analysis.md)。下文保留原预登记。

# 固定 RX 响应的逐包归一化源归因：唯一行预登记

状态 IMPLEMENTED_LOCAL_VERIFIED，尚未读真实因子。上一固定总体均值模型 SSE2.944634、加性.732224、同TX.280647，0/18折及0/5RX改善，已关闭。本行仅判断归一化顺序是否足以解释该差距；不重新拟合 h，不改变旧 FAIL，不视为 IQ 增强候选晋级。

唯一行 `FIXED_RESPONSE_PACKETWISE_ORDER`。输入仅为既有源 signalCoherence 的 L/V180个矩阵、V27000个20复因子和V的rx/day/visibleTX元数据，以及既有 RXresponse 的完整 completion/resolved/result/1800复h。无 raw IQ/source_cache/U/CP/模型/target。原 L6300/U56700 TX隐藏/V27000、全部5RX/day1/2/3、6TX、256点/split392005保持；统计留TX/day均保留全部RX，不是训练域留出。source_ids.json内物理ID已验证，本行不重验、不建立哈希链。

对每个V参考RX1的原300包因子u：z=F u，v=diag(h)z·sqrt(||z||²/||diag(h)z||²)，u'=Fᴴv；零u映射为零并计入300分母。原 mean(u'u'ᴴ) 与旧 Phi_h(mean(uuᴴ)) 比较。90个V矩阵预测仍在5RX中心化，以原V矩阵为读出。全部18折固定原 h、全部20频点、[80,160)窗口、period20、anchor1、referenceRX1、floor1e−12、原控制。0次重新拟合、无幅度／阶数／阈值扫描。V参考因子仅为输入，不更新参数或持久模型状态。

原加性L-only和同TX／其他两日L-only控制不变，复算旧均值预测以核对同一读出。固定资格仍为：加性误差非零；相对加性SSE降低至少20%；至少15/18折改善；全部RX不劣于加性；SSE≤1.05倍同TX控制；继承的固定相位乘积非零。任何条件失败，就关闭这组h的归一化顺序解释，不重拟合、不换窗口/频点/锚点/参考/幅度/阈值救回，不启动IQ训练。即使通过，也仅说明源统计近似偏差，不能证明256点全包滤波身份保持、Ag有效或物理RX替换。

本地21负例、已知共享响应正例、响应反转与TX-only反例、明确不交换例、独立逐包外积对照、零输入和全局复尺度不变性PASS。独立recount端到端及损坏预测/损坏指标反例PASS；36000预测复条目最大独立误差2.086e−17、指标差1.003e−18。检查中单位向量计数字段10修正为8，算法和实测检查未变，不因此重复不变测试。0真实IQ/CP/NN/更新/target。唯一直接P0/P1审查PASS，未发现P0/P1；未访问真实输入/远程或重复测试，不新增审批。

真实主过程：读取180个源矩阵/72000复条目、27000个V因子/540000复条目、1800固定复h；实际使用5400参考V因子，27000次20维因子响应变换，90个V矩阵预测，保存36000复预测条目。独立复算从逐包20×20外积做FFT响应／迹归一化／逆变换，重新计算27000个外积响应、全部90矩阵/18折/5RX，0次拟合。CPU两线程/GPU空，0rawIQ/CP/NN/teacher/epoch/optimizer/头更新/U/target。源身份分数未计算；源统计不能代替最终联合目标。

CWD `PROJECT/releases/cvs_F_RX_response_order_source_20261018_r01`；命令 `CVS-RFFI/bin/python -u -m experiments.cvs_rx_response_order18.run`；唯一输出 `PROJECT/runs/20261018-diagnostic-F-RXresponse-order-sourceCells-manysig-m1-r01`；owner `codex/root/F-RX-response-order-source-20261018`。本地唯一发布入口 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_rx_response_order18.publish`。实际GitHEAD进入不可变release登记，普通账号一次preflight、一次archive本远SHA、remote compile；首步合成/源句柄smoke PASS即继续，无本阶段不用的CP/NN smoke。

只因所属技术异常、非有限、输入角色不匹配或输出冲突停止保留；不自动重试、不改健康旧任务。预期checks/resolved/privatepredictions/result/completion/independentrecount/同row报告。此前设计SHA、成员哈希、签名链、额外review/审批为REJECTED_EXTRA_GATE。未来训练仍F＋跨RX同TX、SIG_TIME/unifiedSinc/joint160、原cosine头联合训练、完整5RX/L6300/U56700hidden/V27000/256点/E30。整体目标ACTIVE。
