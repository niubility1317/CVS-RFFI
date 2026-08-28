# SF-TAPFT P1紧凑部署设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|ENG-01|复盘第七节|区分当前RSS与lifetime max RSS|`sf_tapft_deployment_benchmark.py`、对应测试|verified|115项聚焦测试通过|Linux当前RSS读取`/proc/self/status`；峰值独立命名|
|ENG-02|复盘第七节|冷启动和常驻两种资源基准|benchmark模块、runner、测试|partial|常驻采样已测试|真正逐次新子进程冷启动仍待独立工程候选，不冒充已完成|
|ENG-03|复盘第五节|拆分cache存储/计算dtype/device|adapt、runner、config、测试|verified|配置兼容和负测通过|本轮验证低精度storage一次性转FP32 compute；低精度suffix compute未声明可用|
|ENG-04|复盘第五节|一次性materialize，取消逐步转换|adapt、测试|verified|指针稳定性与重复forward测试通过|每步不再执行dtype转换|
|ENG-05|复盘第八节|独立`CompactH6Suffix`|adapt、测试|verified|logit/gradient等价与引用隔离测试通过|仅D0/D4使用；Q2A/Q2B/R1-T按其真实训练范围走常驻模型delta路径|
|ENG-06|复盘第十六节|delta-only原子写入与失败回滚|runner、测试|verified|写失败保留旧文件、加载失败回滚测试通过|不覆盖合法旧产物|
|ENG-07|复盘第九节|CUDA Graph/AOT/Norm预计算可行性|工程实验行|deferred|独立候选后验证|非D0–D4发布门|
|ENG-08|复盘第九节|冻结suffix eval独立验证|工程实验行|deferred|独立候选后验证|会改变历史H6语义|
|SCI-01|复盘第十四、十五节|Q2A-Deploy|adapt、矩阵config、测试|implemented|503步、mixed norm配置解析通过|实际元素与性能待N607 artifact|
|SCI-02|复盘第十四、十五节|Q2B-Deploy|adapt、矩阵config、测试|implemented|231步、mixed norm配置解析通过|实际元素与性能待N607 artifact|
|SCI-03|复盘第十三、十五节|R1-T support-only温度|adapt、矩阵config、测试|implemented|327步+OOF温度配置解析通过|温度只缩放logit，不改变argmax|
|SCI-04|复盘第十二、十五节|H6后30步head-only class-CVaR|adapt、矩阵config、测试|implemented|可达训练路径与Top2函数测试通过|`lambda_t=0.03`、Top2|
|SCI-05|复盘第十五节|新未暴露capsule D0–D4|config、run报告、N607 artifact|pending|新合法数据句柄尚未定位|先发既有rx20 truth-exposed工程回放；不得据此晋级|
|PRO-01|`项目.md`5.3–5.5|不读source/query/query truth|协议负测、smoke、receipt|verified_local|相关query closure及runner负测通过|远端先做真实checkpoint无query smoke，prediction后再独立scorer|
|PRO-02|复盘第十一、十五节|停止HardPair|config负测、矩阵审计|verified|D0–D4逐行断言`hard_pair_weight=0`|不删除历史实现|
|REL-01|AGENTS最小流程|本地验证、提交、发布、远端核对|Git与run报告|pending|待验证|只stage本轮文件|

当前状态：核心实现与配置已通过115项聚焦测试和唯一一次独立P0/P1审查；两项启动P1均已定点修复并复审PASS。`SCI-05`的新未暴露capsule仍是独立科学证据缺口，不阻断既有rx20上的工程回放发布，但该回放不得用于晋级。
