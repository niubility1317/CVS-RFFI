# D92优化当前交接

- 目标：固定Phase1，全面改善新旧类与各K的H。用户明确强调新类也要好。源域结果不能完成goal；目标分数不能回流调参、选种子或选择性重跑。
- 工作树：`E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt`，分支`codex/d92-support-upgrade-20260928`。根目录不是Git仓库，主承载面其他暂存改动未动。
- 源域旧类诊断180条、注册代理2100条均完成并登记ANALYZED。K1固定FFT权重0；K≥5的support-only选择算法与grid冻结。代理6个TX均被Phase1见过，不能称真正新TX结果。
- 数据run：`20260928-phase2-d92-confirmation-data-manytx-r01`，15444观测、900划分；capsule `residual-noeq-76121e6f34363fa612ec25fb`已VALIDATED_ONCE，不重建或重验。
- r01执行在加载checkpoint前因reference contract没有classes字段失败；无目标预测/评分。保留所有记录。已修复为actual source→注册旧类→ground NPZ类序绑定；15个runner测试通过。
- 已完成运行：`20260928-phase2-d92-scv-confirmation-manytx-m4-r02`，N607 supervisor PID2354788已退出，release `d92_scv_confirmation_20260928_r02`，实际代码commit `841f189690884f0ae51ad8778099fc576e48d849`。完整7236条记录已评分、下载、汇总；不得重复启动。
- 完整矩阵：4model seeds2026092701..04 ×3RX19-1/8-14/8-7 ×3scenes ×K1/5/10/20 ×new0/2/5/10/20 ×5support seeds2026092711..15。每模型900D92+900SCV+9DG，合计7236评分行。RX与当前源/最近目标集合不重合，不宣称所有历史从未使用。
- 当前独立证据：根目录`automation_reports/CV-SincNet/<run>/evidence/readback_1790601598.json`。run已登记ANALYZED；`results/summary/`保留全K、新增类规模、seed与RX/scene、H/两侧准确率/宏F1/floor/forgetting，完整7236覆盖检查通过。4模型compact JSONL/CSV已收集，图已渲染检查。
- 科学结果：候选未通过。每K的Δ旧/Δ新/ΔH（百分点）：K1 +6.46/−7.00/−3.87；K5 +7.82/+0.26/+2.63；K10 +3.00/−0.85/+0.24；K20 −0.23/−0.04/−0.11。不得晋级或标记goal成功，不依据这些目标分数改参数/筛seed/重跑。
- 下一步的范围问题已用async问用户：允许独立源域辅助模型做真正类留出开发，还是禁止所有辅助训练仅用冻结特征。正式4个Phase1均保持冻结。用户回复前不启动依赖此授权的辅助训练。已知源代理六TX均被Phase1见过的局限是目标评分前的证据；辅助训练候选与参数必须只从源域开发选择，禁止将此目标结果传入开发选参流程。
- 已完成不接触目标结果的独立代码可行性核对。具体方案见`docs/D92_SOURCE_AUX_PLAN_20260928.md`：预先固定两组互补3TX、独立scratch辅助模型、TX筛选和连续标签映射、独立辅助导出。状态DRAFT_PENDING_USER_SCOPE，未登记或启动辅助实验。正式6TX源训练实测约19.1至20.0小时/模型，辅助耗时尚未知。待范围答复后再实施依赖步骤。
