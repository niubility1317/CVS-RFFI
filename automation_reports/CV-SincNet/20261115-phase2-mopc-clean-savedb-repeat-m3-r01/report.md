# Clean MoPC 故障同 B 无额外仪表重复诊断

从 full run 失败任务 `0061-hybrid-protonet_cda` 的实际 B-K10，固定 P01/MIX03/14-7 的 clean 新类支持及原 C seed，不加载 query、truth，不生成预测；分别在 GPU 5、6、7 使用原公开仓库公式、无裁剪、无逐层 hook 各重复 10 次。每次由原注册代码从同一 B 深拷贝分叉并固定原 C seed。目的在于检查不同 GPU 与时序下能否重现真实故障，不把无异常称为修复。

入口 `python -m experiments.pamf_phase2_practical.clean_mopc_savedb_diagnostic --runtime <r01 hybrid runtime> --job <failed job> --k 10 --n 20 --device cuda:0 --variant original_uninstrumented --repeats 10`。每卡一个进程、启动前至少 6000 MiB、避让其他 owner。首个非有限停止该诊断分支并记录事件；JSON 中的 status 是实际结果，不以进程退出码代替。输出为独占 `runs/20261115-phase2-mopc-clean-savedb-repeat-m3-r01/gpu{5,6,7}.log`，历史 partial 和 B 不覆盖。
