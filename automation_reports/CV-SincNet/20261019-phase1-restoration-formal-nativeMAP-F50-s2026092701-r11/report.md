# 原生噪声软 MAP：原F50完整七场景确认

状态：LOCAL_VERIFIED，待发布。实际方法 `2b96b5b223e685c2d67b16b7d833ecd61e777150` 已提交并从独立远端HEAD确认可达；并行提交的推送竞争已核实为正常快进，未覆盖其他owner。13项聚焦检查/模块入口通过，候选复用既有R10一次P0/P1，无重复runner审查。N607普通账户预检完成，原source/固定权重可用，新run/release不存在。

本轮新选择仅依据完整R10源结果：native六场景NMSE=0.02760034（R03=0.02836140）；原F50源低城区82.1667%（R03=79.8889%），clean下降0.0556pp，全部六TX低城区改善，最差+0.6494pp。原R10的uncertain仍为身份验收失败，本轮不追认或改写原决定。源身份选择使用V标签，恢复训练0TX标签/0F损失且完整利用U，权限分开披露。

用户验收已固定：lowurban accuracy≥70%；另外五个受损场景分别距同一原F/seed原始clean accuracy≤5pp。恢复后clean损伤单列，不降低参照来达标。候选保持原R10最后E20/4940实际权重，0新训练/适配/拟合头/校准/信道生成。

固定原168000条目标物理记录×七相关视图；三郊区、三城区和clean全部单列。原F50 CROSS_RX-s2026092701/defaulthead冻结；每份IQ的FFT/PA/time都由原F图导出。七视图IQ/features/logits/方差完整、有限和IDs一致后，独立CPU scorer才打开truth。报告accuracy、MacroF1、repair/damage、TX/RX/day、波形/频谱、配对clean特征及用户阈值差距；原R09三种完整读出仅作既有冻结比较，不重跑旧IQ。

source选择和阈值在本轮target前冻结，目标结果不反馈。native组件累计E60、原R03累计E40、训练混合不同；保留R10 direct/physical初值偏差且不应用后续seed修复，不作等成本或纯算子因果声明。单seed、同原生工程代理族，不证明每个补出细节真实或在轨有效。

实际命令/环境/CWD/路径/矩阵/GPU/停止规则和预期artifact见[experiment.json](experiment.json)，需求追踪见[R11设计](../../../docs/PAMF_NATIVE_MAP_FORMAL_R11_20261019.md)。真实2Ls no-query smoke是launcher第一步，通过立即继续；允许GPU0/1/5/7，优先空卡，一条本任务GPUworker、计入全部owner cap2/free12GB，共享capacity4锁覆盖CUDA可见，不停止健康任务。技术故障保留输出、停本row、无自动重试或性能救跑。

发布状态：RUNNING / VERIFIED。单归档SHA比较、远端compile/help、真实2条L_s native＋原F50 smoke通过。实际GPU5 PID/CWD/argv/对应NVML UUID和日志增长核实，完整七场景预测进行中；未提前连接truth或宣称目标达标。
