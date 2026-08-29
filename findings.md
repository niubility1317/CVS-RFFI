# Findings & Decisions

## Requirements
- 根据用户报告实现BiSAGE-D92：阶段A SAGE-D和阶段B SAGE-R均需落地。
- 先完成阶段A最小实验闭环，达到门槛后自动继续阶段B。
- 最终使用完整125验证。
- 必须复用以前D92 E0跑过的数据集、seed配置、split和场景，以便严格同row对比。
- Phase2保持`p2_min_v1`、`VALIDATED_ONCE`、support/query物理样本分离、query只读和truth-last。

## Research Findings
- 历史D92 E0 screening矩阵定义为5个receiver×5个K/新类切片×3组seed=75个identity、225个场景单位。
- receiver为`20-1,3-19,7-14,7-7,8-8`。
- 切片为`K1/new20,K2/new20,K5/new20,K10/new20,K10/new5`。
- 旧screening seed registry为method/support/query `7282101–7282103/7282201–7282203/7282301–7282303`，draw为`7282401`。
- registry另有5组confirmation seed，但用户明确要求历史D92 E0已跑配置，不能直接替换。
- 当前BiNOVA阶段A单行结果没有产生pseudo-D92增益，说明新实现必须补齐类角色外推、梯度共识、协方差一致性和正式D92等价。
- 项目对话索引已刷新到2036条；对`7283101`无历史对话命中，当前仍不能证明confirmation seed被D92 E0正式执行。
- 已找到真正完成的历史`d92_e0_full_only_target125_20260812_v1`：125/125 job、375场景、failed=0，复用原D92 retry2的125个sealed package。
- 历史Target125 receiver为`20-1,3-19,7-14,7-7,8-8`，seed为`713102–713106`，切片为`K1/new20,K5/new20,K10/new5,K10/new10,K10/new20`。
- 历史矩阵manifest为`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`，报告记录SHA256=`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`。
- 可复用历史Target125封装位于`E:/type10-7/code/snapshots/d92_125wt`；builder约26KB、runner约27KB，已具备125 Cartesian覆盖、source package身份校验、truth-free smoke、8-shard和共享技术停止逻辑。
- 当前BiNOVA实现约55KB，已具备基础可微D92、非线性DA/REG模块和四状态生命周期，但缺少报告要求的D92严格等价、类别角色梯度共识、协方差一致性、边界门控与增广拉格朗日旧风险约束。
- 正式D92的`shrinkage="auto"`不是OAS：sklearn先按类用`StandardScaler`标准化，再对每类计算Ledoit-Wolf协方差并按类等先验平均；注册后再以固定0.5/0.5合并旧/新任务协方差。
- sklearn Ledoit-Wolf使用总体协方差（除以N），收缩率由四阶矩`X**2`与Gram平方计算，最终为`(1-beta/delta)*cov+(beta/delta)*mu*I`；要逐logit等价必须在torch中复现该公式及每类缩放还原。
- 优化报告后半部分将最终方法明确为BiSAGE-D92；阶段A前向严格禁止接收类别ID，标签只可用于fit上下文、角色轮换、损失和support cross-fit。
- SAGE-D教师版的锁定机制为：3组4-base/2-pseudo-new覆盖全部6个旧类、每类8/2样本cross-fit、零初始化rank32时间/identity残差、非仿射比例区间、类条件协方差一致性及角色梯度坐标中位数共识。
- 阶段A机制停止条件为预测必须发生变化、非仿射能量至少0.1、held-out pseudo-new的H/min-new/old-post稳定改善；否则属于科学不晋级而非技术失败。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 复用历史D92 E0 feature/package/truth资产 | 保持数据与split完全可比，不触发VALIDATED_ONCE重验 |
| 可微D92先做logit等价测试 | 防止support surrogate改善但正式D92预测不变 |
| 主125比较输出S0/S1/S2 | S0直接D92、S1 SAGE-D、S2 SAGE-D+SAGE-R形成同row因果链 |
| A2–A5和B2/B4/B5用于最小机制行 | 避免把125外层矩阵无必要倍增，同时保留报告要求的机制归因 |
| Target125 manifest保存`source_capsule_id/source_split_id`并逐job比对 | 使新运行不能悄悄改写历史capsule或split |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 最初只定位到后期3-seed D92 screening矩阵 | 通过对话索引和历史Target125报告定位到更早且真实闭合的5-seed E0_FULL_ONLY矩阵；正式方案改用该矩阵 |
| Windows控制台默认GBK导致D92索引搜索输出失败 | 后续设置`PYTHONIOENCODING=utf-8`后重跑 |
| `rg`含空格与管道符的正则再次被cmd拆分 | 改用单token标题检索并按上下文读取，后续不复用该写法 |

## Resources
- `E:/codex/home/attachments/92f5217f-e9ef-40f1-b025-090993f9da67/pasted-text.txt`
- `docs/D92_METHOD_COMPLETE_REPORT_20260727.md`
- `configs/full_ablation_20260728/seed_registry.json`
- `code/cvsrffi/stage2_binova_*.py`
- `code/scripts/run_stage2_binova_d92.py`

## Visual/Browser Findings
- 本任务未使用视觉或浏览器输入。
