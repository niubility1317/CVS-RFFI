# SF-TAPFT-PACE设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|PACE-01|第三、四节|D0作为阶段A和稳定教师|adapt、runner、测试|verified|51项聚焦回归|教师只来自support上的D0状态|
|PACE-02|第四节|稳定样本权重和保持KL|adapt、测试|verified|稳定权重/KL单测|`lambda_preserve=0.10`|
|PACE-03|第四节|阶段B Top2类别尾部损失|adapt、测试|verified|阶段内扩展和loss计数单测|`lambda_tail=0.03`，贯穿120步|
|PACE-04|第三、十一节|E1早层Norm weight、E2全time norm|adapt、config、测试|verified|compact参数映射和矩阵解析|阶段B固定120步|
|PACE-05|第五、八节|4-fold head-only OOF和6参数零和bias|adapt、runner、测试|verified|bias零和、delta v3、clean-single v3兼容|40步，禁止完整模型fold训练|
|PACE-06|第六、十节|按需升级状态机和D0回滚|runner、测试|pending|待TDD|阈值只能来自预登记support规则|
|PACE-07|第七节|按最早可训练Norm选择prefix cache|adapt、测试|verified|logit/梯度等价单测|覆盖`t2.norm`和`time_fuse.1`|
|PACE-08|第九节|OARC时间/参数/存储/内存/计算回执|runner、benchmark、测试|verified|E0常驻3+10次基准|能量/温度无传感器，标记`NOT_CAPTURED`|
|PACE-09|第十一节|E0–E3最小矩阵|config、run报告|verified|4行真实support适配与评分|E0保留；E1/E2/E3拒绝|
|PACE-10|第十一节|新未暴露合法capsule truth-last验证|run报告、N607 artifact|verified|seed713102独立Query120条|四行prediction闭合后才连接truth|
|PACE-11|第十二节|冻结HardPair/Adapter/完整t3/frequency/domain/EMA|config负测|verified|配置解析和许可参数审计|不删除历史实现|
|PACE-12|第九节|能量与热指标|OARC receipt|deferred|N607无星载功率/热传感器合同|只记录`NOT_CAPTURED`，不虚构J或W|

当前计数：verified=10，pending=1，deferred=1，rejected=0，blocked=0。PACE-06的在线触发阈值尚无跨接收机地面冻结依据，因此本轮E1–E3按预登记矩阵强制执行，不从query调阈值；它不阻断验证矩阵。最终E0保留，E1/E2/E3均未通过相对E0科学门槛。
