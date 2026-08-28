# SF-TAPFT-RSE设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|RSE-01|第四、十三节|固定E0结构，仅训练target head和`t3.norm(weight,bias)`|adapt、runner、配置、测试|verified|58项聚焦回归、独立P0/P1审查|禁止E1/E2扩展、类别bias、HardPair、Adapter、full t3、frequency/domain更新和EMA|
|RSE-02|第五节|构建原始视图与一个保守增强视图的prefix cache|adapt、测试|verified|相位旋转幅度保持、双视图cache和物理样本计数测试|0.05rad为本轮预登记`UNPUBLISHED_DEFAULT`，不读取query|
|RSE-03|第五节|双视图训练增加JS一致性，固定`lambda_view=0.05`|adapt、测试|verified|loss可达、双视图4步训练测试|LOO-proto只读原始物理视图，避免同一物理样本的增强副本泄漏|
|RSE-04|第六节|两折分层cross-fit重复两次，选择`step={250,350,450,520}`与`alpha={0,0.25,0.5,0.75,1}`|adapt、runner、测试|verified|单轨迹snapshot、缓存validation suffix和同轨迹提交回归|每个fold只训练一条520步轨迹|
|RSE-05|第六节|support-only稳健风险：MacroCE、class-CVaR、view JS和margin regression|adapt、测试|verified|风险分量与保守tie单测|选择规则不得读取query或truth|
|RSE-06|第六节|若适配风险未优于`alpha=0`，选择DA0；否则全support重训并提交单一插值delta|runner、bundle、测试|verified|alpha插值、DA0 tie回退、同轨迹selected snapshot测试|最终仍为单bundle、单推理路径|
|RSE-07|第七节|两个类别均衡子模型的许可delta均值，并在全support上低学习率收尾30步|adapt、runner、测试|verified|共同anchor、delta均值、strict clean-single回归|首轮R3固定每类8条、两个子模型|
|RSE-08|第十一节|记录full forward、suffix forward/backward、head step、视图数和FBE输入计数|audit、runner、测试|verified|receipt字段、真实selection与validation cache计数|逐模块MAC/FBE在缺少MAC基线时标记`NOT_CAPTURED`|
|RSE-09|第九节|发布R0、R1、R2、R3最小矩阵；R4仅在R1或R2单独通过后运行|配置、run报告|verified|R0–R4全部artifact和评分闭合|R1通过后才触发R4；R4不作为首轮前置gate|
|RSE-10|第九、十二节|使用新seed713103、旧6类K=10、最大独立Query120，truth-last同row评分|run报告、N607 artifact|verified|5行prediction receipt与详细scorer|本轮不注册新类|
|RSE-11|第十节|满足时间、参数、delta、cache和内存资源合同|runner、benchmark、报告|partially_verified|GNU time、RSS、bundle、cache及R0–R3 dmon|R4完整GPU峰值和10次常驻P90未捕获，不冒充已验证|
|RSE-12|第十二节|按域报告配对BA/floor/NLL；跨域统计升级留待多域矩阵|报告|deferred|单域结果只能支持方向性判断|receiver bootstrap、三场景和K退化曲线不扩大本轮最小矩阵|

当前计数：verified=10，partially_verified=1，deferred=1，rejected=0，blocked=0。RSE结构已严格落地；0.05rad增强幅度属于报告未给出具体数值的`UNPUBLISHED_DEFAULT`。R4已按条件闭合但不晋级；跨域统计仍未完成，不冒充稳定性结论。
