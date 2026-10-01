最新只读证据：[evidence/query_runtime_1790863040880987200.json](evidence/query_runtime_1790863040880987200.json)（2026-10-01 13:56:19 UTC），一行失败，其他两行实时运行，一行等待原排程；整体未完成且未评分。

## 当前：一行数学梯度失败，其他行继续运行

状态仍为 RUNNING，但完整矩阵已经出现技术失败。2026-10-01 13:24:09 UTC 独立读回：rx3/s2026092702 的 PID 869448 已退出（返回码 1）；实际根因是 C 阶段 inner head 的 `UNSUPPORTED_ACTIVE_JACOBIAN`：全部紧约束不等于独立且严格互补的 working set。supervisor 报告缺少 `predictions_complete.json` 是子进程失败后的二级现象，不是文件传输失败。

supervisor 869328、rx3/s2026092701 的 869442、随后由原排程启动的 rx1/s2026092701 的 944965 均有实时进程证据；rx1/s2026092702 仍等待排程。证据见 [evidence/query_runtime_1790861110983221800.json](evidence/query_runtime_1790861110983221800.json)。这里只读进程、状态与失败文本；未读取预测数组、query truth 或分数。

保留全部已有产物，不重试失败行、不改变当前不可变 runtime、不停止健康进程。全部四行完成的评分前提未满足，不能评分部分行后当作完整结果。A/B/C 准确率、H 和改进幅度仍为 N/A。本次故障仅允许数学求导与求解正确性的源码修复，不能据此查看 query 成绩选参。

<!-- row_failure_observed_1790861110983221800 -->

以下保留此前预登记与历史观察；实时状态以上述证据为准。

## 当前：完整重复基准已唯一启动并独立核实

状态RUNNING，实际runtime d86edc3235ec0ca0e1925223970e07373256fb9c，supervisorPID869328/start10047966，实际argv/CWD/CPU两lane及BLAS两线程均独立VERIFIED。最新证据[evidence/query_runtime_1790853755995305700.json](evidence/query_runtime_1790853755995305700.json)：rx3两行PREDICTING（PID869442/start10049751与869448/start10049849），rx1两行PREFLIGHT_COMPLETE等待原supervisor排程；全部四行preflight通过，尚无整体完成或query成绩。

既有Git目的地已通过认证账户/仓库权限和项目origin核实，自动审批允许原Git推送；远端OID=d86edc3235ec0ca0e1925223970e07373256fb9c，ahead/behind=0/0。不是用户新增回答，也未换用其他写入路线。独立source交付见[evidence/source_delivery_20261001.json](evidence/source_delivery_20261001.json)。

原冻结方法及资源不变，support分析成绩不回流。全2400parent/旧6/完整K×新增类×4receiver×3scenario×5support seed。只使用本次legal support重拟合B并真实继承到C；query逐样本全注册类只读。全部四行A/B/C固定后才单独scorer；此处是已评分库存复用，不是新独立验证。

以下为保留的预登记和历史状态；运行判断以上述当前证据为准。

# MarginJoint 完整三阶段重复基准预登记

状态：PLANNED，尚未发布或启动。源代码已合成验证；Git push 的具体目的地授权待用户确认。原 support 数学分析继续，方法参数不从结果反馈调整。

固定 Phase1 与 practical residual/post_sync/noeq/25MHz。使用 support 实验开始前已固定的两个最小模型种子，不按成绩挑选种子。rx3 的 900 个 split 与 rx1 的 300 个 split 全部保留，共 2400 个 parent。旧类 6；K=1、5、10、20；新增类=0、2、5、10、20；3 场景、全部 4 接收机及 5 个 support seed。六类 seed 逐行见 experiment.json；support 为五值轴，原 row 标量为 null。

A 使用冻结地面原分类头；B 仅用当前旧类 support；C 从该次实际 B 继承后注册新类。三阶段使用同一物理旧 query；C 面对全部已注册类统一竞争。K1 正常评估；新增 0 的新类准确率/H/差距为 N/A。报告完整 K×新增类矩阵和接收机、场景、模型、support seed 分层，保留逐 parent 记录。

继续禁止源样本、源逐记录特征、query 拟合、truth/role/配额/全局重排和评分回流。已有 p2_min_v1/VALIDATED_ONCE capsule 与 source-only final200 来源不变；不重建/重验数据，不加载 encoder/checkpoint。地面小包只保留原分类头及元数据，字节数与实际传输分开。

模型来源沿用已核实的完整 scratch/final200 契约，逐行绑定 checkpoint SHA 与原 cache 来源。同 model 跨 cohort 的原目录允许不同，checkpoint 与 Ground packet 保持一致。原 support-probe 的适应状态不继承。冻结配置、完整逐行 argv、环境/CWD、输出/log/预期artifact见 experiment.json 和 spec。

资源使用原先明确的4096 transitions/167772160 factor bytes，仅为技术保护，不保证收敛或进程内存。CPU 两 lane、BLAS 两线程、GPU 禁用。仅技术失败影响所属 row；不重试、覆盖或停止健康任务。保留详细训练文本、完整结构化事件/状态及紧凑 JSONL/CSV；实际参数、耗时、RSS、常驻状态、传输未知项记 N/A。

全部 row 的 A/B/C 固定并独立读回后才单独连接 truth。旧 BranchLocalRidge 仅配对相同两个种子的同 capsule/split/query；缺失 A/B 不补造。理想目标是适应提升≥10个百分点、注册后旧类下降≤1个百分点、新旧类绝对差≤3个百分点，作为软目标。此处是已评分数据的透明重复基准，不能称为新的独立验证。

独立入口审查未发现 P0/P1，见[审查文档](../../../docs/D92_MARGIN_JOINT_QUERY_ENTRY_REVIEW_20261001.md)。root 的实际激活 ssr-gpu 元数据 preflight 核实4行、冻结公式完全相等和原 QP 额度，未拟合或访问真实输入。真实 PID、runtime OID、完整预测和成绩仍为 N/A；publisher 仅在源码已授权推送后执行一次。
