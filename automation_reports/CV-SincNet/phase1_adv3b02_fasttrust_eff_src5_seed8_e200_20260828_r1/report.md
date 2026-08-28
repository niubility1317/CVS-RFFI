# Phase1 ADV3B02+FastTrust-EFF SRC5八种子扫描报告

## 最小预登记

- Run ID：`phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1`
- 当前状态：未启动；本地实现与唯一一次独立P0/P1审查完成，待Git发布和N607预检。
- 研究目的：在`SRC5_MAXP2`源域上比较同种子的ADV3B02从头训练对照与FastTrust有效子集，使用源域`V_select`冻结星地信道性能最好的FastTrust种子。
- 数据协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；四个角色均仅来自源接收机，物理样本两两不交；目标接收机数据不参与训练、校准、选模、候选重排或选择性重跑。
- 源域profile：`SRC5_MAXP2`；源接收机`1-19,18-2,19-2,2-19,3-19`，ManySig索引`1,3,4,6,8`；目标接收机`1-1,14-7,2-1,20-1,7-14,7-7,8-8`，ManySig索引`0,2,5,7,9,10,11`；源/目标接收机严格不相交。16行扫描只构建源接收机索引`1,3,4,6,8`的4日数据及源域`V_select`，不构建或访问目标接收机加载器；目标4日数据只属于冻结后的独立确认阶段。
- 初始化：两臂均从头训练。旧`ADV3B02_CORE90_SOFT_E200`checkpoint含`1-1,14-7,2-1`源数据，而这3个接收机在新profile中属于目标域，因此禁止作为本轮初始化。
- 矩阵：8个种子`713101`至`713108`；每张GPU绑定一个种子，并并发运行`ADV3B02_CONTROL`和`ADV3B02_FASTTRUST_EFF`，共16行、每卡两行。
- FastTrust-EFF：保留高可靠三头一致硬伪标签、类平衡上限、源域先验对齐和高可靠U卫星身份CE；取消固定50%身份回填、soft/candidate身份路由、时序稳定门、prototype证据、U prototype更新、cross-RX损失和nuisance损失。
- 训练预算：E200、U batch256、eval batch512、fused student forward；源域重评E1起每10epoch一次，最后20epoch每epoch一次；Sinc前端保持FP32数值路径。
- 选种规则：只读取每行最终checkpoint的源域`V_select`。主分数为`source_val_sat_hmean=H(clean,min(leo_clear_weak,leo_low_elev_weak,leo_rain_weak))`。FastTrust种子只有在相对同种子对照满足`LEO mean`提升、`LEO floor`不下降且clean下降不超过0.5pp时才可进入一次性目标确认；多个通过种子按主分数、LEO mean、LEO floor、clean、种子升序依次打破并列。若无种子通过，则记录负screen，不声明FastTrust提升，也不使用目标域反向重排。
- 目标确认数据：只在源域冻结最佳种子后核对`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`、`split_id=260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25`；预测先形成artifact，独立scorer随后连接truth。目标结果不反馈选种、调参、重训或重跑。
- 本地环境/CWD：`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\phase2-canonical-union-maxq`。
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release目录待Git提交后冻结。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1`；禁止覆盖已有路径。
- 日志：run root下`dispatcher.log`、`dispatcher_logs/<candidate>.log`和每行`train.log`。
- 预期artifact：每行`final_ssdg.pth`、`config.json`、`status.txt`、`metrics_joint.json`，以及clean和3个LEO弱场景各自的metrics/log。
- 技术停止规则：仅协议/query越权、错误seed/split/receiver/day、错误checkout、输出碰撞、无法启动、无prediction闭合、同一确定性训练前异常至少出现在两行、进程归属不清或可能影响无关任务时停止；性能低不停止。
- Git提交、release路径、精确启动命令和GPU实时占用将在本地验证与N607预检后回填。
- `REJECTED_EXTRA_GATE`：单独设计审批、spec SHA、成员级hash、额外seal/receipt和重复数据验证不属于排他最小实验工作流，本轮不执行。

## 需求追溯

|ID|来源|要求|目标文件/产物|状态|验证|备注|
|---|---|---|---|---|---|---|
|`SRC5-ISO`|`项目.md`与新数据profile|源/目标接收机不相交，扫描阶段不得混入目标域|`code/dataset_wisig.py`、`code/SSDG/train_ssdg.py`|本地已实现|聚焦协议负测已通过；真实ManySig smoke待N607|默认旧跨日行为保持不变；扫描模式`target_access=false`|
|`FT-EFF`|历史FastTrust同row机制证据|只保留硬路由、类平衡cap、源先验和U卫星身份CE，禁止低可靠回填|`code/cvsrffi/muse_ssdg.py`、worker launcher|本地已实现|route单测已通过；远端launcher dry-run和真实checkpoint smoke待N607|不使用依赖污染旧checkpoint的RC4/anchor cache|
|`SEED8-PAIR`|用户当前请求|8个种子，每卡同种子对照与FastTrust各一行|matrix+dispatcher|本地已实现|矩阵结构测试已通过；N607 GPU/PID检查待执行|共16行|
|`SPEED`|用户当前请求与既有速度证据|U256、eval512、稀疏源域重评、fused forward|matrix+worker launcher|本地已实现|静态命令与聚焦测试已通过；运行日志计时待N607|不牺牲E200或最终四场景评估|
|`SOURCE-SELECT`|`项目.md`|只用`V_select`选种，target预测后独立评分|matrix+本报告+分析脚本|已冻结|配置回读+完整artifact分析|目标结果不得反馈研发|
|`NO-OVERWRITE`|`AGENTS.md`|不可覆盖run root并只停止本run进程树|dispatcher|本地已实现|远端dry-run+真实启动绑定检查待执行|不得使用广泛`pkill`|

## 本地实现与验证

- 代码实现：新增源域专用评估模式、显式4日接收机不相交保护、FastTrust硬伪标签不回填路由、scratch配置去除未使用checkpoint记录、16行双实验调度器及冻结矩阵。
- 聚焦验证：共享日期/接收机隔离负测、源域`V_select`唯一加载器、FastTrust硬路由、矩阵两行/GPU、协议测试、严格星地评估、速度配置和Sinc数值测试均通过；Python编译检查与`git diff --check`通过。
- 独立审查：唯一一次P0/P1审查覆盖训练器、FastTrust路由、数据构建器、worker、seed-scan launcher、矩阵、测试和两份报告，结论为未发现P0/P1问题；未增加白名单外gate。
- 本地Git Bash探针未获得`MSYSTEM=MINGW64`，因此未把WSL替代通道用于launcher验证；N607发布后执行远端`bash -n`、行为dry-run及真实checkpoint无query smoke。
- 根目录报告位于非Git目录；Git镜像为`automation_reports/CV-SincNet/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/report.md`，提交时使用精确强制stage以越过仓库对`automation_reports/`的默认忽略规则。
