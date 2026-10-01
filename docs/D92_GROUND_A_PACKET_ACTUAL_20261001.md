# 适应前原地面分类头：实际小包导出

ARTIFACTS_COMPLETE/VERIFIED：r02实际runtime0145234，supervisor617603正常退出；两个源分类头包经独立loader、checkpoint/class/scale/eps/source lineage/文件统计读回核实，实际文件8737B与8735B，其中各3840B为原float32权重。两包已下载并核对metadata/complete及各文件字节。query/support/源样本/源逐记录特征0，encoder未构造/执行，A仍未评分。ground export程序内计时0.306050687s，进程峰值RSS450482176B；不含Python/Torch启动耗时，不代表星载资源或链路传输，实际星地增量传输N/A。

完整模型与类别来源、原生类型兼容、失败r01与实际r02见[run报告](../automation_reports/CV-SincNet/20261001-phase2-d92-ground-a-packet-m2-r02/report.md)。原头行序为14-10、14-7、20-15、20-19、6-15、8-20，s=30、eps=1e-4，raw feat_joint z_id160。导出不改模型或任何训练状态，不把R0/B0当A，不读取目标样本。

适应前A与B−A仍N/A，真实小包已准备好；后续独立配对补充不会覆盖原方法summary。固定A逐记录面对全部六个原头类别，先持久化预测，后连接合法held support真值，并与实际B/C的同物理旧held样本配对。K1无独立held时N/A，query评分与星载资源另行报告。
