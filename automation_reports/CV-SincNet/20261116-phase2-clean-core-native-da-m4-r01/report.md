# Clean 主对照：原生域适应 K10 缺项

用户将 clean 对照限定为 P01–P03、MIX01–MIX08、七接收机、四模型种子及 K5/K10 × N5/N10。已评分结果保留并优先复用。当前核实原生六种域适应方法各有五个 seed 2026092701、P01/MIX01、接收机 1-1、14-7、2-1、20-1、7-14 的 K10 A0 与 B 预测缺失；各方法 K5 A0/B 已有，不重新预测。本队列只补 30 个 K10 任务，每任务两条 A0/B；任务接口占位 N5 不表示对域适应方法执行新类注册。

输入为 `20261112-phase2-cleanpaired-data-r01` 与原三池 capsule，目标 support/query 均 clean。源 checkpoint、数据角色与 `20261113-phase2-clean-native-pilot-r01` 对应 profile/seed 的 runtime 一致；各 worker 开始时执行真实源 checkpoint、无 query smoke。目标旧类 K10 支持用于 B；A0 不拟合目标，两个阶段均只在预测固定后由既有独立 truth-last 服务逐任务评分。

唯一新 run `20261116-phase2-clean-core-native-da-m4-r01`；配置准备目录同名；代码身份为包含本报告的 Git commit。计划由 `clean_core_native_da` 从精确匹配的 pilot 任务与只读缺项清单生成，不启动已有任务，不输出其他 K/N。发布前对全部 clean scheduler 的固定/部分预测及潜在活动 owner 按身份检查，并单独核实真实资源；旧 MoPC owner 已自然排空。

N607 允许 GPU 0/1/5/6/7，每 GPU 本队列最多 7 worker、启动前剩余显存至少 6000 MiB，逐次显存快照最多新增一 worker，其他 owner 不干预。系统性技术故障停止新增派发并保留输出；不改变其他方法。运行命令为 release 内 `python -u -m experiments.pamf_phase2_practical.clean_scheduler dispatch --config <同名计划目录>/scheduler.json`，预期每任务有两份预测、独立评分、日志和完成标记。自动评分延续已运行的逐任务 CPU scorer。
