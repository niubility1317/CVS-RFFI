# CVS 跨复数通道相干读出：源域实验预登记

状态：PLANNED。原无增强基础网络目标恢复，本轮普通交叉熵唯一、无星地增强、仅身份骨干，最终只测clean。此前六环境测试属于用户插入的独立评估，不用于本轮结构、参数或选模。已有未发布的数学原型在公共输入上指出同通道功率／滞后相关对逐通道常相位不敏感；本轮检验显式跨通道相干是否改善合法源和clean表现。

核心仍是energy_equivariant的六个共享RMS复数block。新readout保留logpower均值／标准差与lag1／2复相干，用与固定channel0、floor(C/2)的复相干实／虚部替换四段位置功率。每通道10量，读出维度、参数、深宽不变，202553参数。零新增参数相对同核心energy对照，原residual_fusion164225参数另作性能基准，不称与其等参数。

rho(c,a)=mean(z_c conj(z_a))/sqrt(mean|z_c|²mean|z_a|²+1e-6)。共同常相位抵消，跨通道相对相位保留；固定anchors不依赖包、TX、RX或query集合。RX／信道同样影响特征，绝不解释为唯一TX硬件；舍弃位置功率和低能量anchor可能损害性能，负结果保留。候选crossphase_raw保留原received输入；crossphase_half另有预登记固定半量相对CFO校正，alpha不是TX／RX晶振贡献比例。完整数学、通信、物理及RFF论证与原论文链接见[前瞻设计](../../../docs/CVS_CROSSPHASE_IDENTITY_HYPOTHESIS_20261002.md)。

两个候选各4seed=8个scratch模型，model／loader seed2026092701至2026092704；固定split392005、L／U／V=6300／56700／27000，U不用，sourceRX1、3、4、6、8／day1、2、3，六TX／equalized1／center256／unitRMS。无初始化／resume／teacher／EMA／原型继承。复用既有数据VALIDATED_ONCE，在实际构建时核对全物理角色。AdamWlr2e-4／wd1e-4／cosine1e-6，batch128保留末批，E200×50=10000步／模型；完整FP32，cuDNN和matmul TF32false；无裁剪、域骨干或任何额外loss。

四份既有energy_equivariant源控制只读取源配置、契约、scratch来源、完整曲线与E200指标，绝不加载权重。其为此前固定源性能规则赢家，实时preflight核实E200完成，无目标选模。在8新＋4控制全部源记录齐全后，按4seed mean(0.5源V＋0.5最差源RX)最高优先，性能完全并列后才比较成本。所有权重固定E200，不以最佳轮或公共物理误差选模，目标成绩不回流。

若新候选选中，独立冻结配置后发布4行新clean预测并复用24行既有冻结clean控制，全部完整预测固定后独立truth-last。若energy控制保留，则沿用其原已完成clean结果，新候选测试记N/A，不读未选query。预登记测试run为20261002-phase1-cvs-crossphase-clean-manysig-m28-r01；每row clean168000／6类／7目标RX，原capsule/index/truth路径见experiment.json；不新建数据、LEO、support或新类，不重新测试历史控制。

完整stepJSONL、epochJSONL、compactJSONL／CSV和详细文本保留实际CE／权重、LR、全梯度、执行旗标、源V／最差RX、alpha、六block共享能量诊断、实际固定anchor相干、耗时／峰值显存。冻结公共级联检查常相位与部分波形频率协变、TX／RX非可辨识反例和源90个TX×RX×day单元；全部诊断不更新模型、不参与源排名、不虚构硬件参数。资源测量在丢弃clone上执行，不回流训练。

57项初始聚焦检查通过；collector导入边界修正后17项sourceprotocol检查通过，独立P0/P1审查收敛；8个本地丢弃CPU模型完成24个CE更新。远端仍将使用CVS-RFFI已验证Python做冷启动无数据检查，正式模型先做真实checkpoint往返无query smoke，再进入原source数据。

一个launch owner／独占新输出；每GPU最多2个总训练实验，按实际可用容量发布。保留所有历史产物；不停止、重启或热改其他任务，无低性能停机或自动重试。源指标和合成性质通过均不构成测试提升或原目标完成。

[逐行登记](experiment.json) · [一次preflight](evidence/preflight.json) · [聚焦检查](evidence/local_validation.json) · [本地CPU检查](evidence/local_cpu_smoke.json)。

## 实际发布与启动

状态RUNNING／VERIFIED。发布commit `ded20eebd01f66d1411bc16384cbad0c32029df8`，N607远端CPU验证PASS，独立读回8个实际训练进程、独占输出、GPU0至7及实际source resolved参数，全部已完成至少3轮且日志增长。仅CE、scratch、无增强、identity-only、50步/轮及完整FP32实际生效。此时尚无E200选模或clean结果，不能称性能优化完成。详见[evidence](evidence/running_readback.json)。

## 本轮交接

最新独立读回8进程均存活，E93至E99，无失败；健康任务保持不变。条件clean入口已实现并通过相关95项回归、新扩展25项检查及一次独立P0/P1审查，两组测试有重叠，不合称120项。source-selected新候选只生成4份clean预测，原24份固定控制只读复用，共28行先预测后评分；energy控制保留则新候选测试N/A。当前没有新clean成绩，性能目标继续保持未完成。

[最新进程与日志证据](evidence/latest_source_readback.json) · [评估入口验证](evidence/clean_path_validation.json) · [下一步交接](handoff.json)。
