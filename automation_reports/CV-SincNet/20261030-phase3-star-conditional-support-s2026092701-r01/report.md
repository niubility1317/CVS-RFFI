# PAMF 第三阶段：预测身份条件局部支持诊断

## 预登记（2026-10-30）

Run `20261030-phase3-star-conditional-support-s2026092701-r01`，实际执行 code `0e5cfae40fb7f91825f695fbf4a1ac4fdaa6f0fe`。固定一个 original_F0 row、两分数 pooled_relative 控制和唯一主 predicted_identity_relative；k8/eps1e-6/FP64逐行unit/clip(2−2cos,0,4)/mean8，128查询分块。主方法仅把known邻域限制为原默认头预测c对应的source L真旧TX bank，OE仍全bank；身份始终原F0 headargmax，不使用query真TX/RX/场景/K，不更新bank或query群体统计。

方法、数据与接口见 `docs/PAMF_PHASE3_STAR_CONDITIONAL_SUPPORT_PLAN_20261030.md`、`configs/pamf_phase3_star_conditional_support_20261030.json`。完整Phase3用户v3仍跨新RX星地C_K/R_U≥90%、同输入原F0closed−A_K≤2pp，主报trueTX宏A_K/R_U/H；本轮只源端机制证伪，不降低全目标。

复用local-support r01/code84cdc012f7d9641b283c407f6048bfe4e6e21cf6的同G10 MIX01 P1 originalF0单位fused160/logits：L6300/OE6134 bank、V27000、novel3150，合42584同物理记录。source五RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、2021-03-08/15/23均已见；旧6TX14-10/14-7/20-15/20-19/6-15/8-20，L各1050/V各4500；novel19-15/19-16/7-1各1050已有源开发曝光，不能作未见RX/TX盲测。新增仅cacheL已有6300个y元数据与IDs对齐，用于source bank分组；OE仅outside-six成员监督，OE真TX/K禁止。sourceL标签和V校准是额外已使用的源监督，非全系统零源标签。

原F0固定Phase1三视图SSL E50/11100/232993/default6head、modelseed2026092701/sourcecontract392005（原L6300/Us56700/V27000，训练池10%有标签），不继承r02更新后的F。bank原teacher仍r02/code5aa041122396d48c49b8cb3d5912649bdc3217e1在增量前由originalF0生成；原DURA v1.0 R69/266951/FP32/TF32off/reuse+graph身份保留。本轮F/head/G/optimizer更新0，真实IQ值消费0，真实F/G前向0；额外F2仅首步actualcheckpoint synthetic G10P0 smoke，不是新G/P1端到端测试。父F30152与历史P1恢复成本保留祖先，非零总成本。

bank先于V固定，控制raw/六τ/margin/接收精确复现父pooled_relative，控制不重新校准。主只V标签经既有predicted-class complete-tie风险规则，保护42组（overall+5RX+6scene+30交叉）trueTX宏CK90/原baseline损2pp，新增6τ；全部12τ/cal/known写全后才打开父novel特征。预测完整后独立CPU scorer先验证公有来源/默认head/新2分数/父控制/τ，再读private source novel真值，不重拟合或做父全三score重验。

**完整逐样本source bank仅作source诊断，不能带入部署。** 即使主known42及unknown整体/5RX/6景12组90%科学门槛通过，也仅READY_FOR_SOURCE_FREE_EXPORT；eligible_primary/target_eligible固定false，后续只含冻结聚合锚点或参数的source-free候选须另记。失败NO_PROMOTION，不换主/调k/扩seed/目标调阈值。本轮targetU/Q/正式RX24-18/原Us新读取0，DA/发现/注册0，AN/HOW=N/A，fullgoalACTIVE。

### 执行与预期产物

- Host N607，Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-conditional-support-s2026092701-r01-release`，唯一owner `codex-phase3-star-conditional-support-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_conditional_support_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-conditional-support-s2026092701-r01-release/runtime.json`。
- 父特征 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-local-support-s2026092701-r01/rows/original_F0/predict`；cacheL metadata/ids `/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache/L`；原F0 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/final_ssdg.pth`，契约 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F/main_split-repl-shuffled_time-3view-s2026092701/source/source_contract.json`。仅复用VALIDATED_ONCE角色句柄，不读真实IQ数组，不增加hash/receipt。
- run root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-conditional-support-s2026092701-r01`，row `rows/original_F0/predict` 与 `score`、configs/logs/state各独占不可覆盖；本地 `E:/type10-7/local_artifacts/pamf_phase3_star_conditional_support_20261030_r01`。source启动第一步smoke PASS立即继续，一个实际free≥4096MB GPU predictor、随后独立CPU scorer；每卡最多2新增，不停改其他owner。
- source输出bank7fields、known/novel原单位features/logits/defaulthead与两个raw/margin、12τ/cal/complete/smoke；score results/complete、actual configs/launch/pipeline/rowstate、完整日志/observations。private truth仅CPU scorer `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz`，仅在新prediction complete与全部公有校验后连接。
- stop仅来源/角色/权限错、非有限、输出碰撞或无法完成合法prediction，保留partial；性能差不停止/重启/选择性重跑。启动结果不明先只读对账同run，不重复提交。

主报宏AK/RU/H/CK、原baseline损伤、42/12分组与逐TX/RX/scene、AUROC/OSCR/CI、同ID配对。NN499944=F232993+G266951，无aux；bank FP32特征7957760B/FP64工作15915520B，L_y50400B与RX/IDs/索引/workspace另计。新生产CPUscore和独立复核分别报告，不冒充B1端到端时延。

验证：19 passed, 408 warnings, 57 subtests passed in 6.72s；四模块AST/实际runtime PASS，一次独立P0/P1 PASS。覆盖条件邻居/控制数值一致、标签权限/默认头、smoke先source、bank和12τ冻结顺序、私有truth在公有验证后、PASS也非targeteligible及source→score终态；不重复父suite。单preflight确认父完成/原F0身份及sourceL合法句柄/GPU余量，真实IQ/target读取0；单code/config commit、一个archive比较/compile、actualcheckpoint smoke、PID/CWD/cmd/GPU/log与全预测后独立评分，不加未来exporter或报告审批。
