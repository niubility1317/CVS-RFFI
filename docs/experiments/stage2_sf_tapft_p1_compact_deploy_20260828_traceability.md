# SF-TAPFT P1紧凑部署设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|ENG-01|复盘第七节|区分当前RSS与lifetime max RSS|`sf_tapft_deployment_benchmark.py`、对应测试|verified|118项聚焦测试通过，N607 GNU time实测|Linux当前RSS读取`/proc/self/status`；峰值独立命名|
|ENG-02|复盘第七节|冷启动和常驻两种资源基准|benchmark模块、runner、测试|partial|常驻采样已测试|真正逐次新子进程冷启动仍待独立工程候选，不冒充已完成|
|ENG-03|复盘第五节|拆分cache存储/计算dtype/device|adapt、runner、config、测试|verified|配置兼容和负测通过|本轮验证低精度storage一次性转FP32 compute；低精度suffix compute未声明可用|
|ENG-04|复盘第五节|一次性materialize，取消逐步转换|adapt、测试|verified|指针稳定性与重复forward测试通过|每步不再执行dtype转换|
|ENG-05|复盘第八节|独立`CompactH6Suffix`|adapt、测试|verified|logit/gradient等价与引用隔离测试通过|仅D0/D4使用；Q2A/Q2B/R1-T按其真实训练范围走常驻模型delta路径|
|ENG-06|复盘第十六节|delta-only原子写入与失败回滚|runner、测试|verified|写失败保留旧文件、加载失败回滚测试通过|不覆盖合法旧产物|
|ENG-07|复盘第九节|CUDA Graph/AOT/Norm预计算可行性|工程实验行|deferred|独立候选后验证|非D0–D4发布门|
|ENG-08|复盘第九节|冻结suffix eval独立验证|工程实验行|deferred|独立候选后验证|会改变历史H6语义|
|SCI-01|复盘第十四、十五节|Q2A-Deploy|adapt、矩阵config、测试|analyzed_no_promotion|Q180 BA 84.4444%，floor 53.3333%，NLL 0.519486|类4相对D0回退10pp，floor和类别保护失败|
|SCI-02|复盘第十四、十五节|Q2B-Deploy|adapt、矩阵config、测试|analyzed_no_promotion|Q180 BA 82.2222%，floor 53.3333%，NLL 0.511932|BA和floor失败|
|SCI-03|复盘第十三、十五节|R1-T support-only温度|adapt、矩阵config、测试|analyzed_no_promotion|Q180 BA 85.5556%，floor 63.3333%，NLL 0.575789，58.45秒|类4保护、NLL和时间失败；只保留性能研究档|
|SCI-04|复盘第十二、十五节|H6后30步head-only class-CVaR|adapt、矩阵config、测试|analyzed_pass_no_incremental_value|Q180与D0同为150/180，NLL仅改善0.000699|分类argmax完全不变，时间增加2.09秒|
|SCI-05|复盘第十五节|新未暴露capsule D0–D4|config、run报告、N607 artifact|pending|新合法数据句柄尚未定位|先发既有rx20 truth-exposed工程回放；不得据此晋级|
|PRO-01|`项目.md`5.3–5.5|不读source/query/query truth|协议负测、smoke、receipt|verified_n607_truth_last|真实checkpoint smoke PASS；Q60/Q120 prediction完整后评分|receipt确认适配阶段未打开query/source，score为truth-last|
|PRO-02|复盘第十一、十五节|停止HardPair|config负测、矩阵审计|verified|D0–D4逐行断言`hard_pair_weight=0`|不删除历史实现|
|BUG-01|P1运行定点审计|Compact梯度裁剪必须覆盖优化器全部参数|adapt、回归测试、D0/D4 r2|verified_replayed|强裁剪reference/compact轨迹`atol=1e-7`；r2完成Q180|旧D0/D4标记方法不匹配，不作为有效P1结果|
|BUG-02|真实checkpoint smoke|低精度storage必须先materialize为FP32 compute|smoke、测试、D0/D4 r2|verified_replayed|smoke最大logit/gradient差均为0|r1技术停止且无性能结果，r2使用新run ID|
|REL-01|AGENTS最小流程|本地验证、提交、发布、远端核对|Git与run报告|ready_for_final_publish|release SHA一致、远端编译PASS、结果已闭合|最终报告提交后独立核对远端OID|

当前状态：核心实现与配置已通过118项聚焦测试、20项部署/benchmark测试和唯一一次独立P0/P1审查；两个真实运行问题均已定点修复并用新run ID重放闭合。D0是通过完整门槛的最小候选；D4通过但没有增量价值；D1/D2/D3不晋级。`SCI-05`的新未暴露capsule仍是独立科学证据缺口，不阻断既有rx20工程回放发布，但该回放不得用于正式科学晋级。
