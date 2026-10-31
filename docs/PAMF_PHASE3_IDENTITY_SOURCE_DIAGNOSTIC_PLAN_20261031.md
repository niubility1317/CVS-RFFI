# Phase3 身份初始化与保真源端诊断（2026-10-31）

Run `20261031-phase3-identity-source-diagnostic-s2026092701-r01`。上一轮共享学生未获晋级，且没有合并初值的查询读出。本轮只检查合法源端，区分 r03 骨干与原分类头初始衔接、源训练后的变化；不补目标初值测试，不根据已曝光 Q 调参。完整 Phase3 目标继续 ACTIVE。

固定比较三个全部 source-only 的模型：原 F0、r03 body/projection＋原 F0 W 的未训练 JointIdentity 初值 INIT、上一轮 SOURCE_JOINT444。严格复用 f540 联合训练配置和原 source-only 祖先，SOURCE 的 U 读取为0；不加载 TARGET_JOINT 或其他 U 适配权重。F0、r03、SOURCE 和 DURA 全冻结，优化/适配/阈值/新密度拟合为0。

输入仅既有 source G10 P1 L6300/OE6134，保持源五RX三日及原物理角色；L真标签本来合法，OE只保留成员身份和公开RX/day，不读其真实TX。V/NV/原U_s/目标U/Q/任何private truth读取为0。每模型对每条P1恰好一次编码；F0原生头与INIT/SOURCE的同次raw_fused160默认头分别计算logits。DURA真实前向0，复用已验证P1；首次实际F0/r03/G无query两条合成smoke后立即继续诊断。

报告源 L 的原生默认头 macro 准确率/CE、逐 TX/RX/day、原正确变错/错变对，和 OE 的最大 softmax/熵等无 TX 诊断，不伪造未知宏拒识率。场景分组仅使用该角色现有 metadata；L 没有 scenario 时保留 N/A 和原因，不重算或猜测场景，不为补分组读取额外数据角色。比较 F0/INIT/SOURCE 的逐 record 原生融合特征方向与范数、INIT/SOURCE 的单位 z160 漂移、默认 W 变化及 source 每 TX 跨 RX/day 质心的类内/类间几何。只做这些统计，不训练或部署新的身份头、原型分类器、旋转映射或适配器，不给出目标性能推断。

必须验证 INIT W与原F0逐值相同、INIT除W外与source-r03严格一致，SOURCE与其source-only444 checkpoint一致；默认head的源端差异不得被当作未见接收机因果。L/OE是模型已训练角色，源训练诊断不是独立测试。所有模型参数/buffers在统计前后保持，输出不得覆盖旧run或回流改旧候选。

唯一owner root；N607单GPUworker，实际空余≥4096MiB、每卡本任务新增≤2，外来进程不动。新代码commit、聚焦负测/一次独立P0P1、最小登记、一次preflight/单archiveSHA和compile、实际PID/CWD/fullargv/GPU/log增长及source-only完成标记组成执行证据；无truth-side scorer，因为不读取private真值，L只是合法训练标签统计。

真实编码预算F012434、INIT12434、SOURCE12434，共signal24868＋F012434；G真实0。合成F0/r03/G各2，若增加模型合成核对按实际独列。只生成源端诊断记录/汇总，目标预测0、科学性能结论NONE；任何新方法另行预登记固定源选择规则和全三DURA测试，不能从本行宣称Phase3已达标。
