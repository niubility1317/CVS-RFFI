# D92优化当前交接

当前用户纠正（2026-09-28）：辅助训练禁止使用源域数据；允许地面已冻结模型、原型及少量汇总统计，结合卫星合法support。暂无硬传输上限，优先小payload并报告字节数。源TX留出辅助模型方案已被替代，未启动任何新训练；其实现草稿已归档并从正式代码撤出。当前推进source-data-free Phase2方法，禁止读取source IQ、逐样本特征、source loader或由原型重建伪源样本。允许的量化摘要按当前`项目.md`5.3.2及实际冻结来源判断；无需再次询问辅助训练授权。下列此前“等待范围答复”条目只记录历史。

新候选D92-SFHead-v1与冻结配置见`docs/D92_SOURCEFREE_AUX_20260928.md`及`configs/d92_sourcefree_frozen_20260928.json`。仅support上的taskbalancedCE+旧类KD+L2，CPU确定性L-BFGS-B，无source校准、无query选参。新增地面统计0字节，现有ground摘要只读核算8191至8383字节/模型。37项core/summary/allocation/predictor合成测试、42项runner/scorer测试通过，独立P0/P1审查无阻断。新run：`20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01`；data run：`20260928-phase2-d92-sfhead-data-manytx-r01`；release：`d92_sfhead_confirmation_20260928_r01`。RX20-19按未用于本轮评分且完整26TX可用性选取；4模型×1RX×3scene×4K×5newcounts×5supportseed=1200配对split，连同基线和DG共2412记录。每TX180物理记录（3scene各supportpool30/query30），全矩阵固定后才独立评分。

当前已VERIFIED启动：supervisor PID2403988，实际release commit `3348c8c37df402098a6f84397eaa1bf74d8d592e`。`readback_1790604450.json`核实4个D92子进程的实际argv/CWD一致，4模型已完成checkpoint无query推理检查及received特征提取。数据已ARTIFACTS_COMPLETE，capsule `residual-noeq-d0a99fede324159c5a4750fd`，4680观测、300splits，VALIDATED_ONCE。保持此run，禁止重复启动或改运行中配置。监控命令：`python tools/read_d92_run.py --spec configs/d92_sourcefree_confirmation_20260928.json --compact`。新结果尚未读取，完成后下载scores并用支持新spec的汇总器评分；不能以本地测试或启动成功标记goal完成。

本地汇总器已支持`--spec configs/d92_sourcefree_confirmation_20260928.json`，20项合成测试通过；由spec验证2412条精确矩阵覆盖，按每K的ΔH>0、Δnew>0、Δold>=−0.01判断。本地spec只补充了与发布前data_config及notes一致的`data.scenarios`和机器可读acceptance字段，未修改远端配置、候选或矩阵。终态确认后可用`read_d92_run.py --download scores.json complete.json startup.json`下载，用`summarize_d92_confirmation.py --scores <path> --spec <spec> --output <new directory>`汇总，`collect_d92_fit_logs.py --spec <spec>`收集全部compact JSONL/CSV；保留远端完整fit_trace。旧`finalize_d92_confirmation_record.py`写死SCV旧run，不可用于本轮。需新增本轮独立算术核验与最终报告，不能误写旧结果或晋级。

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
