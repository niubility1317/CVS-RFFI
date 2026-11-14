# Phase 2 clean MoPC 修复版接续

从已失败并排空的 `20261114-phase2-mopc-public-clean-full-m4-r01` 接续。六类 B 主干之一 ProtoNet→MoPC 的二次输入桥替换为明确标记的 `proto_affine_v1`；其他 MoPC 主干及原生 MoPC 保持原版。新桥保持旧类实际支持判决、softmax 与中心化 logits，去掉不必要的平方范数输入；仓库 MoPC 损失与 SGD 不变、不裁剪。新桥改变 C 参数化，旧 ProtoNet→MoPC 结果保留为历史版本，不与新版合并成同一方法。

原队列 432 项完整、1 项旧桥非有限、其余未启动。已完成的非 ProtoNet→MoPC 不重算；已完成的旧版 ProtoNet→MoPC 需用同一次实际 B 状态运行新桥 C，作为新版本而非重算同一版本。旧故障项在新版本和新输出中复用已保存的 B-K1/K5/K10；只有旧任务未运行到的 K 才可拟合新 B，旧 partial 与失败日志保持。无旧 B 的未启动任务正常拟合各自 B。所有复用绑定 capsule、seed、K、split、源及数据，父 run 必须失败排空；桥接版本写入状态与输出标签。

入口 `python -m experiments.pamf_phase2_practical.clean_mopc_affine_resume --parent-config <r01 scheduler.json> --plan-root <project>/preparations/20261115-phase2-mopc-clean-affine-resume-m21-r01 --run-root <project>/runs/20261115-phase2-mopc-clean-affine-resume-m21-r01 --release-root <immutable release>`。先完整核验祖先、生成计划、验证并原子登记唯一后继，再使用 GPU5–7 每卡最多7、启动前至少6000 MiB，避免其他 owner。任一故障停止新增派发、排空并保留输出；全局评分由 r03 等待全部保留组及新版 MoPC 固定后独立触发。真实故障首个算术根因仍未证明，仿射候选已通过相同 B、12 个不同 B/K 支持集检查以及已知二次桥合成失败案例，但不保证其他 MoPC 主干没有独立故障。
