# 轻型DA×精简D92联合小筛选追踪

设计源：`docs/STAGE2_RD_GOAL_20260731.md`§1、§4、§5、§8、§10

当前状态：`GOAL_REVISED / DESIGN_FROZEN / CORE_LOCAL_VERIFIED / RELEASE_PATH_INCOMPLETE / NO_NEW_PERFORMANCE_RESULT`

本表只追踪会直接影响方法正确性、因果可识别性或下一次真实小筛选运行的项目。D119-CFO真值门、D102/D124全量回放、重复数据验证、通用authority、额外receipt和报告美化均为非阻塞P2，不进入本表。

|ID|目标要求|目标文件或工件|状态|完成证据|是否阻塞小筛选|
|---|---|---|---|---|---|
|JD92-01|最多3条原理不同的DA候选共享一个精简D92；浅/中/晚层不是连续层扫描，不开发3个head|联合设计卡、`stage2_d127_da_candidates.py`|verified|提交`45485b18`；A/B/C候选与共享head聚焦测试通过|否|
|JD92-02|梯度型DA采用Phase1物理隔离`S_src→a¹→Q_src`元目标；Phase2只更新低维状态，基础模型与封存方向不更新|`stage2_d127_phase1_assets.py`、`stage2_d127_checkpoint_hooks.py`|verified|真实checkpoint bridge、部署qKNN outer loss、query零更新测试通过|否；真实资产生成另见JD92-14|
|JD92-03|无反传晚层候选与梯度候选保持独立机制；不叠加A+B或临时组合|`stage2_d127_da_candidates.py`、checkpoint hooks|verified|候选C独立summary/hypernetwork路径及置换测试通过|否|
|JD92-04|精简D92删除formal288维管线的old/new角色分裂、重复稠密拟合、D62行拼接及无独立贡献的FFT96/RF32块；全注册类标签置换等价；与formal D92的差值只称全管线替换差|`stage2_d127_d92_lite.py`|verified|K5解析OAS-form对角LDA、INT8/FP16状态、类置换测试通过|否|
|JD92-05|K5形成真实D92-Lite交互；K1若统计不可辨识则显式qKNN边界，不伪造类内方差|D92-Lite状态与联合screen|verified|K1逐值alias、K5非alias、两臂/四臂一致性测试通过|否|
|JD92-06|同一160维空间冻结核心`2×2`：`M0/M_DA/M_L92/M_JOINT`；`R_D92_FORMAL`只作公共同row全管线参照；`M_DA_D92`仅S1胜者可选诊断|`stage2_d127_joint_screen.py`、`stage2_d127_s0_entry.py`|verified|公共臂18次、适应臂54次和完整三候选矩阵测试通过|否；formal历史参照连接另见JD92-15|
|JD92-07|base/adapted各只做必要forward；qKNN和D92-Lite共享规范化z160、support索引和query缓存；formal参照独立复用历史artifact，不冒充同head|checkpoint hooks与S0入口|implemented|本地forward/cache receipt与跨候选base逐值相等已覆盖；真实S0运行receipt待生成|是；只缺文件入口实证|
|JD92-08|S0固定`seed=713102`、receiver`{20-1,3-19,7-14}`×K1/K5×3scene=18行；S1为剩余receiver18行；先完整prediction后评分|目标、冻结设计、S0 typed row|verified|提交`fec8c14b`、`3d07db6e`；18行覆盖/顺序/完整性测试通过|否|
|JD92-09|S0仅保留3个方向条件：DA的`ΔH>0`、K5 Lite-after-DA的`ΔH>0`、联合`ΔH>0`且总正确数增加；其余指标报告不设0.5pp硬门|分析规格|verified|目标§5与冻结设计§6已冻结|否；只决定结果后晋级|
|JD92-10|相对formal Target D92：单平面INT8`B_lite=164C`、C26状态减少74.1%，联合状态≥50%缩减、K5拟合MAC≥90%缩减、拟合时间≥50%缩减、query head MAC减少44.44%；另报DA端到端资源|D92-Lite与S0 resource receipt|implemented|解析字节/MAC公式和代码计数已测试；同机墙钟与端到端资源待S0|否；不阻塞S0，阻塞最终胜出|
|JD92-11|本地只做协议负测、真实checkpoint无query smoke、聚焦单测和独立`P0=0/P1=0`；不新增控制面|6个D127测试文件、独立复审|verified|`ssr-gpu`下50项通过；A/B/C、D92-Lite、四臂、Phase1 bridge、S0复审均`P0=0/P1=0`|否；新文件仍需一次独立复审|
|JD92-12|本地Git提交和不可覆盖run报告后，由唯一Terra Max runner执行N607发布；Luna不执行SSH或实验|Git、run报告、runner handoff|implemented|核心与目标提交已完成；run报告和新入口提交待完成|是|
|JD92-13|复用D92 retry2的既有K1/K10/K5包；生成逐class/scene的K5⊂K10有序opaque physical-ID前缀及同query根紧凑receipt，不重验数据|D127 S0输入适配器及测试|pending|matrix manifest已预登记prefix；runtime逐项receipt尚缺|是|
|JD92-14|从Phase1 source received-IQ和合法标签构建7个receiver-held fold；每个receiver×class固定前5 support/后9 query且K1⊂K5，按冻结128步日程训练A/B/C，以循环标签置换替代42折class-LOCO，量化、parity、只读落盘并重载|Phase1文件builder/asset wire/CLI及真实工件|pending|核心训练函数只接受内存episode；尚无真实有标签episode builder、wire或资产工件|是；当前最高风险|
|JD92-15|把sealed target enrollment/apply package转换为18个truth-free`D127S0Row`，加载checkpoint/资产/qKNN lock，独占写出完整预测，并在预测封存后连接独立truth scorer和历史formal D92同row参照|最小S0 CLI、package adapter、scorer及测试|pending|现有S0仅为in-memory API|是|
|JD92-16|创建不可覆盖run ID、报告、精确命令/GPU/路径/stop rule和同步映射，独立复审新入口后交给唯一Terra Max runner|本地报告、Git提交、runner handoff|pending|N607只读资源盘点完成，尚未sync/launch|是|

## 结果后停止语义

- 某候选完整小筛选不满足方向性双收益：`COMPLETED_DIAGNOSTIC_NEGATIVE / CLOSE_CANDIDATE`；
- 三候选均负：保存完整证据，转向新的方法原理，不在层、rank、步数、view、shrinkage或阈值上调参复活；
- 仅胜者进入剩余2receiver的S1；S1失败不递补runner-up；S0/S1均不得冒充Target性能；
- 协议或确定性执行错误才允许技术停止；中间性能弱不得停止正在运行的完整冻结矩阵。

## 实施与发布待补

1.只实现JD92-13至JD92-15所需的最小文件路径，不重建D92数据或通用控制面；
2.先生成并重载三份真实Phase1量化资产，再运行18行S0；
3.对新增文件执行聚焦负测、真实checkpoint smoke及一次独立`P0=0/P1=0`复审；
4.在run报告冻结不可覆盖run ID、精确命令、GPU、路径和stop rule，Git提交后交给唯一Terra Max runner。
