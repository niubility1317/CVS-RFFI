# SF-TAPFT P1紧凑部署设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|ENG-01|复盘第七节|区分当前RSS与lifetime max RSS|`sf_tapft_deployment_benchmark.py`、对应测试|pending|待测试|Linux读取`/proc/self/status`|
|ENG-02|复盘第七节|冷启动和常驻两种资源基准|benchmark模块、runner、测试|pending|待测试|每条资源行隔离|
|ENG-03|复盘第五节|拆分cache存储/计算dtype/device|adapt、runner、config、测试|pending|待测试|兼容旧配置|
|ENG-04|复盘第五节|一次性materialize，取消逐步转换|adapt、测试|pending|待测试|P0B慢路径根因修复|
|ENG-05|复盘第八节|独立`CompactH6Suffix`|adapt、测试|pending|待测试|不得持有完整model引用|
|ENG-06|复盘第十六节|delta-only原子写入与失败回滚|runner、测试|pending|待测试|不覆盖合法旧产物|
|ENG-07|复盘第九节|CUDA Graph/AOT/Norm预计算可行性|工程实验行|deferred|独立候选后验证|非D0–D4发布门|
|ENG-08|复盘第九节|冻结suffix eval独立验证|工程实验行|deferred|独立候选后验证|会改变历史H6语义|
|SCI-01|复盘第十四、十五节|Q2A-Deploy|adapt、矩阵config、测试|pending|待测试|不超过1248元素|
|SCI-02|复盘第十四、十五节|Q2B-Deploy|adapt、矩阵config、测试|pending|待测试|不超过1368元素|
|SCI-03|复盘第十三、十五节|R1-T support-only温度|adapt、矩阵config、测试|pending|待测试|不得改变argmax|
|SCI-04|复盘第十二、十五节|H6后30步head-only class-CVaR|adapt、矩阵config、测试|pending|待测试|`lambda_t=0.03`、Top2|
|SCI-05|复盘第十五节|新未暴露capsule D0–D4|config、run报告、N607 artifact|pending|待实验|同row最大合法Query|
|PRO-01|`项目.md`5.3–5.5|不读source/query/query truth|协议负测、smoke、receipt|pending|待验证|prediction后独立scorer|
|PRO-02|复盘第十一、十五节|停止HardPair|config负测、矩阵审计|pending|待验证|不删除历史实现|
|REL-01|AGENTS最小流程|本地验证、提交、发布、远端核对|Git与run报告|pending|待验证|只stage本轮文件|

当前统计：`verified=0`、`deferred=2`、`rejected=0`、`blocked=0`、`pending=14`。

