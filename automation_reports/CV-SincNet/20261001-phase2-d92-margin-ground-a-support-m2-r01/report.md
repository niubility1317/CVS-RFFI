# MarginJoint三阶段配对：独立Ground A补充预登记

状态：PREPARED_NOT_LAUNCHED_WAITING_FOR_VERIFIED_MARGIN_SUMMARY。尚未启动，不运行训练或校准。唯一launch owner=root。

单一Margin来源20261001-phase2-d92-margin-joint-support-m2-r01，实际训练runtime e50de0ad4e7d0570dce62ba88c3791fdf1c14951。两个固定source-only模型seed×两个capsule，共四行/160parent；旧6类、K1/5/10/20、新增0/2/5/10/20。数据与原Margin物理划分完全一致。

A使用已核实的原地面float32分类头packet；不加载checkpoint或encoder，不用源域样本/逐样本源特征/query。每次A预测先固定，再由独立scorer连接合法support truth。B/C只取当次Margin已固定预测；不拟合、重置或拼接其他方法状态。

启动前须原四行完成且唯一独立Margin数学摘要为COMPLETE_MARGIN_JOINT_PROBE_VERIFIED，runtime/source身份一致。当前尚无完整摘要，不能发布或评分；不增加数据重验或审批。

三阶段旧类指标严格同物理held配对，C面对全部注册类别竞争。报告A、B、C_old、C_new、H、B−A、B−C_old和绝对新旧差，保留40个diagnostic×K×new cells与receiver/model分层。K1无独立held=N/A，R0不是A。理想10/1/3个百分点是soft目标，非逐轮门槛。

一条CPU lane/BLAS2，四行顺序评分，每row独占目录。记录实际程序墙钟、RSS、packet/数组文件字节；缺失星载实测和传输字节=N/A。仅support持出诊断，不称独立query或新增数据确认。结果不回流训练、选模、参数或重跑。

原始输入、训练release和已有输出保持不修改。失败保留，无自动重试；本补充不干预健康Margin任务。

[预登记](experiment.json) · [来源绑定](evidence/prereg_binding_20261001.json)

已核实可继续使用既有已发布源码：独立源码副本 `E:\type10-7\code\snapshots\d92_published_ground_20261001_clone` 的 HEAD 为 `8d020384f11a89228aabc744630a3efed11b98f1`，11个所需源码/配置的 canonical Git blob 与当前版本完全一致；Windows checkout 换行规范化后文本也一致。该副本不含尚未推送的新 query 入口。仅 Ground A 补测可在原完整数学摘要生成后按既有 publisher 执行；启动时仍核实真实远端分支 OID、输入和独占输出，不能根据此准备记录宣称已启动。新 query 推送继续等待具体 GitHub 披露目的地授权。
