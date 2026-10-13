# 频域特征提取：旧伪标签门控＋EMA＋cos

目标为 GLOBAL_FREQ 固定三种子 clean 平均准确率至少 84%，当前尚未取得新实验性能结果。三组 LEGACY50、LOCAL_FREQ、GLOBAL_FREQ，各用 2026092701/02/03 三个独立模型种子，共 9 个从零模型。另一个引用对话改造时序与 PA 卷积块；本矩阵只改变频域 f1/f2 的特征提取。

用户已明确旧门控指历史伪标签门控＋EMA＋cos，原实现为 experiments/cvs_legacy_no_sat/design.py 的无星地增强配方。此轮保持 RX/day 自适应置信门控（quantile0.86、tau0.92至0.97）、batch_neighbor（window2、min_conf0.8）、强增强一致门控、EMA0.999（versioned update、无 startup averaging）、lambda_u0.16、lambda_ent0.01、label_smoothing0.01 和历史普通源增强。没有卫星训练、U_s TX 真值、目标训练或目标选模。旧 VAT 中 CE50 的 79.5411% 是纯 CE 历史控制，不再称为此历史半监督基线。

全部模型固定 50 轮×222 步＝11100 次更新，共 99900 次更新；第1至32轮 label，第33至50轮 pseudo，cosine50 lr0.0002至0.000001，AdamW weight_decay0.0001。原200轮130＋70压缩为50轮32＋18，在新目标访问前固定；不冒充原200轮总成本。源 V 只作健康诊断，不选模、不设候选晋级门槛。固定最后第50轮 checkpoint；所有预登记行自动 clean168000 个预测，预测固定后独立 truth-last 评分，共1512000个预测和162条同row分层指标。

频域原输入为32个镜像配对频带的四类功率/不对称特征。f1池化输出16通道×16位置，f2池化输出32通道×8位置。LOCAL_FREQ与GLOBAL_FREQ分别在这两层后添加相同的 packet-local residual attention+FFN，仅注意力可连接位置不同：局部±1（不跨首尾），全局全位置；heads4、qkv width2C、FFN4C、固定频带索引sin/cos、残差scale0.01。两组新增参数各21248，身份参数198273，原结构177025。freq_gate、pa_gate、时序/PA分支、池化/投影/融合及普通cosine6×160头保持。新增模块独立CPU初始化，保留外部CPU/CUDA RNG，配对初始骨干/头权重相同。

方法依据 [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) 的跨位置加权信息交互思路；这里是频段适配，并非该论文RF实验复现或已证实的物理不变性。现有特征压缩/局部路径只是可检验结构假设，不宣称已定位性能瓶颈。

数据沿用 ManySig 既有契约和 VALIDATED_ONCE capsule。L_s6300、U_s56700（TX标签隐藏）、V27000，源RX1/3/4/6/8、day1/2/3、6TX、split392005，目标clean168000固定物理ID；NoExplicitCFO输入策略全组匹配。源角色和checkpoint来源核对，实际data/query权限不变，不重验既有IQ或追加哈希/allowlist链。当前随机模型不加载E20/E200或任何外部teacher；EMA由本row自己创建。真实E20只作一次无查询烟测，不继承到训练。该基准已在历史研究中暴露，不声称全新独立盲测或星地部署。

本地结构/梯度/远段依赖、逐包独立、同参数控制、头与原骨干权重保留、EMA新增参数版本更新、CPU/CUDA RNG保留、5条合同负测已PASS。训练producer→source_proof→native checkpoint恢复回归三组均PASS，27条错误metadata拒绝；U TX隐藏检查PASS。隔离归档路径导入和随机新结构构造PASS。唯一独立P0/P1审查与真实checkpoint无query smoke按最小流程执行。没有读取源IQ、目标IQ或truth进行这些本地检查。

实际代码/config身份为release_commit.txt内的单一Git commit。发布命令：verified Python native runner experiments.cvs_frequency_context50.publish --output local_artifacts/frequency_context_delivery；远端执行：CVS-RFFI Python -u -m experiments.cvs_frequency_context50.dispatch。环境与CWD为N607普通账户下新release root，路径详见experiment.json。release/run不可变，每GPU最多3个总实验进程（包括其他owner），共享容量锁，空闲至少12000MiB。归档一次本地/远端SHA比较、远端compile、一次预检、真实无query smoke PASS即调度。

首个本run技术失败停止新调度，健康行完成；保留全部partial，不盲重试、不热改、不影响其他owner。不按源或目标性能停止/重排/调整/重跑。训练后每row及时预测和独立评分，唯一launch owner为本主Agent。预期模型、source/config/freeze、完整step/epoch/stdout、prediction/complete、accuracy/F1/confusion/RX/day/TX、analysis与completion及同row报告。

用户再次明确每轮实现和对应实验结果上传Git：发布前代码/config/预登记commit和push，完成后结果/实际成本/失败或负结果另commit和push，均独立对比远端OID。数据、权重、原始日志/预测及本机状态不纳入Git。

REJECTED_EXTRA_GATE：brainstorming的另行设计/审批前置与项目排他最小工作流冲突，本授权目标内的可逆结构选择直接执行；只保留本最小报告，不创建额外审批链或report-only审查。

唯一独立 P0/P1 审查 PASS：旧伪标签门控/EMA实际启用，第33轮pseudo，新增参数进入optimizer/EMA；50轮11100更新与预测衔接一致，9行配置匹配，发布依赖齐全，源性能早停/卫星/目标训练关闭。审查只读，无源/目标IQ或truth访问。

发布 VERIFIED：执行commit43bd20494aa470568a68553ce664dc76811b9707，远端OID已核对一致。单归档SHA传输比对、远端compile及真实E20 no-query smoke PASS；dispatcher PID1540996。9个worker实际PID/CWD/argv/GPU匹配，全部日志增长，当前第12至14轮，错误队列为空，每GPU总进程最多2（上限3）。实际resolved args确认50轮32label+18pseudo、旧batch_neighbor/strong gate、EMA versioned0.999、label_smoothing0.01、cosine50；CFO显式坐标关闭，全组匹配。性能结果仍等待固定训练、预测、独立评分完成。

终轮故障核验：9 行均已完成50轮/11100更新并自然退出。原生checkpoint参数类型依赖baseline_origin_sat_view在torch.load前未导入，预测入口统一失败；没有目标prediction或score。原产物保留。另行登记20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01，同9个终轮权重、零追加训练、原clean矩阵不变。
