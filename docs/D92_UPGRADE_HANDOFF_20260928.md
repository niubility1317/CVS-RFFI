# D92优化当前交接

- 目标：固定Phase1，全面改善新旧类与各K的H。用户明确强调新类也要好。源域结果不能完成goal；目标分数不能回流调参、选种子或选择性重跑。
- 工作树：`E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt`，分支`codex/d92-support-upgrade-20260928`。根目录不是Git仓库，主承载面其他暂存改动未动。
- 源域旧类诊断180条、注册代理2100条均完成并登记ANALYZED。K1固定FFT权重0；K≥5的support-only选择算法与grid冻结。代理6个TX均被Phase1见过，不能称真正新TX结果。
- 数据run：`20260928-phase2-d92-confirmation-data-manytx-r01`，15444观测、900划分；capsule `residual-noeq-76121e6f34363fa612ec25fb`已VALIDATED_ONCE，不重建或重验。
- r01执行在加载checkpoint前因reference contract没有classes字段失败；无目标预测/评分。保留所有记录。已修复为actual source→注册旧类→ground NPZ类序绑定；15个runner测试通过。
- 当前运行：`20260928-phase2-d92-scv-confirmation-manytx-m4-r02`，N607 supervisor PID2354788，release `d92_scv_confirmation_20260928_r02`，实际代码commit `841f189690884f0ae51ad8778099fc576e48d849`。已独立核实4个checkpoint特征导出完成和4CPU配对预测启动；不重复启动。
- 完整矩阵：4model seeds2026092701..04 ×3RX19-1/8-14/8-7 ×3scenes ×K1/5/10/20 ×new0/2/5/10/20 ×5support seeds2026092711..15。每模型900D92+900SCV+9DG，合计7236评分行。RX与当前源/最近目标集合不重合，不宣称所有历史从未使用。
- 当前独立证据：根目录`automation_reports/CV-SincNet/<run>/evidence/readback_1790599606.json`。读回工具`tools/read_d92_run.py --spec configs/d92_confirmation_recovery_20260928.json`，精确ssr-gpu Python；Windows原生powershell.exe，禁pwsh/WSL。
- 下一步：只读观察现有run。只有root/complete.json为SCORED且全部行完成才下载`scores.json complete.json startup.json data_reuse.json`，执行`summarize_d92_confirmation.py --scores <download> --output <new-summary-dir>`；完整7236覆盖检查、每K和K×new_count、seed与RX/scene、H/两侧准确率/宏F1/floor/forgetting全部报告。汇总已有6个合成测试通过。
- 完成后运行`collect_d92_fit_logs.py --spec ...`提取4模型的compact JSONL并生成CSV，不含query大数组。登记ANALYZED，镜像报告与相关证据，显式stage、commit、push及远端OID核对。
- 如果候选未达到全面目标，如实报告。不能从这批目标成绩改参数后重跑、筛选seed或删除失败结果。未取得完整性能证据前goal保持active。
