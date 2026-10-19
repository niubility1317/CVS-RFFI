# 接收观测恢复第二轮

状态：RUNNING / VERIFIED（六行源训练与独立评分阶段）；方法代码`7efdba92932a86bf4b0492a565d35377564a26da`。12项聚焦检查、实际原生full/residual配对/旧residual重放、条件训练梯度与完整采样检查通过；一次独立P0/P1复核无阻断问题。

本轮落实用户两份新报告。固定六行、数据/seed/命令/成本/输出/停止规则唯一维护于[experiment.json](experiment.json)，详尽设计和逐项状态见[设计追踪](../../../docs/PAMF_RECEPTION_RESTORATION_R02_20261019.md)。

同一个增强前源观测分别进入原生full与residual；同seed、几何、初始状态和注入RX，不连续叠加。仅post_sync/noEQ，每epoch新信道流。三行同U-Net分别用15%clean+85%res、15%clean+85%full、15%clean+60%res+25%full；后三行在同混合流上比较复时频、多尺度条件扩散、流匹配。六行均scratchE20/4940更新、全部6300L/56700U，单model seed。数据生成16CPU worker；GPU0/1/5各最多一条本run worker，其他owner计入cap2/free12GB。

第一阶段是纯配对自监督。训练不读取TX标签、F预测或F特征，不使用PL/CE；F只在冻结后读出。基本目标IQ NMSE+0.1稳定化log-power差，生成模型保留自己的速度目标。无场景真值路由，clean同样经过R。新增噪声比、模拟锁定和状态仅用于源分层日志，不输入R；扩散时间与物理退化分别记录。

源选择A仅用原坐标波形误差；B明确使用有标签source V身份资格，属于标签辅助选模。两者在目标前冻结，最多测试两个去重候选。旧1200条固定目标输入原字节复用，不新抽信道，不追逐目标分数。其结果只是单seed阶段性配对诊断，尚不能证明完整正式lowurban>60%或稳定身份恢复。

原F成本另计：cleanF纯CE E200/10000更新；AsF scratchE20+自身E10/6660更新。R的每行E20有1260000次物理曝光，不与F的epoch成本混同。有限L身份保护、联合微调、粗恢复+残差生成、真实算子展开和pre_sync均未启动，保留后续匹配对照要求。

首步controller执行真实已有F两源样本no-query smoke并直接继续；首个真实训练更新进行R checkpoint往返检查。之后每row自动训练→最终权重冻结→clean+六residual预测/独立评分→同latent full源交叉诊断→条件shuffle/单样本延迟。全部源行完成后冻结选模，再做目标预测/独立truth-last与同row报告。

技术异常按登记规则保留输出、阻止未启动行、让健康活动行完成，不自动重试。release/run均不可覆盖。已有READY/EXACT_MATCH源缓存、父source V与F来源结论复用，不重建数据验证链。仅一个新代码归档与一次传输比较；不重新传数据或权重，不增加设计审批或签名gate。

## 发布与实时状态

方法commit`7efdba92932a86bf4b0492a565d35377564a26da`已推送，远端分支OID独立读回一致。新release和run独占，单个代码归档传输比较通过，远端compile/import通过。实际环境Python3.10.19、Torch2.1.0+cu121、NumPy2.2.5，未改包。数据与F权重复用既有路径，不重复传输或验证。

独立进程证据确认controller存活、CWD与完整argv匹配本run。20个epoch共享full/residual源缓存已全部生成，6300 L_s与56700 U_s、0 TX标签/目标读取；F真实源样本与前三个R真实checkpoint往返检查PASS。前三行均完成E20/4940更新、126000 L及1134000 U曝光与双路源独立评分；其余复时频、扩散、流模型已自动启动。GPU0/1/5各一条本run worker；其他owner任务保持。

这不是训练/评分完成状态。后续以相同controller继续，禁止因观察超时重启或重新生成已完成缓存。实际运行证据保存在本地私有产物目录；数据、权重、原始日志、账号路径和进程完整快照不提交。

## 首批源域结果（中期）

仅本run source V的1800条固定样本、原residual/post_sync/noEQ面板。原AsF默认头clean为98.7778%、low urban为59.9444%；所有R在完全冻结后读出，不重新训练或替换头。三行全20轮训练记录和完整stdout均已解析，无技术错误标记，训练TX标签读取为0。

|行|AsF clean（%）|AsF low urban（%）|low urban变化（百分点）|low urban NMSE|
|---|---:|---:|---:|---:|
|unet_residual|98.6667|64.3333|+4.3889|0.120908|
|unet_full|98.6111|64.7222|+4.7778|0.122031|
|unet_mixed|98.6111|64.0556|+4.1111|0.121703|

三行出现源身份正向信号，clean损失0.1111至0.1667个百分点。完整六行结束前不冻结候选，不依据此中期结果改变活动训练或矩阵。标签身份诊断不等于无标签波形选模；尚无本run目标分数、跨接收机结论或最终>60%证明。完整中期汇总见[源结果](results/source_interim.json)。
