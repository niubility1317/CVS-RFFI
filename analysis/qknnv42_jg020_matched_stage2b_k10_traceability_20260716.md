# JG_R8_LR020严格配对Stage2-B K10追踪表

目标：在与ADV3B02三域适应对比矩阵相同的5个target receiver、5个seed、K=10物理support/query ID和单Query View策略下，运行锁定的`JG_R8_LR020`，形成可与MRIOR-SDA、DADDA-SDA和ProtoNet CDA配对比较的旧类域适应证据。

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|JG-M01|用户|覆盖相同5个target receiver×5个seed|计划生成器、worker、报告|local_verified|25行矩阵清单与完成回执|receiver=`20-1,3-19,7-14,7-7,8-8`；seed=`713101..713105`|
|JG-M02|用户|固定K=10|配置、candidate lock、worker|local_verified|所有row的`k_shot=10`|不在本轮扫K|
|JG-M03|用户|复用相同support/query ID|离线package gate|implemented_pending_remote_evidence|逐receiver×seed按顺序比较JG包与旧三方法K10 split manifest|不使用query标签或性能选ID|
|JG-M04|用户/项目.md 8.5|相同推理View策略|TTA policy、predictor|implemented_pending_remote_evidence|query固定单View；历史support/query种子公式和runner哈希绑定|JG固有support-only三场景增强单列计算量|
|JG-M05|用户|使用锁定`JG_R8_LR020`|runner、candidate lock|local_verified|`joint_gate/rank8/alpha8/lr0.02/5epoch/50step`硬校验|包含既有P4 ground adapter|
|JG-M06|项目.md 7.1|Phase2严格LEO_weak-only且clean不可达|离线cache/package、Landlock、访问审计|implemented_pending_remote_evidence|pre-open SHA/成员审计、runtime allowlist、strace访问审计|Phase2配置不暴露dataset路径|
|JG-M07|项目.md 7.2|无query角色Oracle、类别数量/配额或全局分配|predictor、split manifest、测试|local_verified|五个协议字段、truth-free prediction、独立scorer|禁止Hungarian/OT/dense query graph|
|JG-M08|项目.md 10.2|输出旧类适应指标|scorer、汇总器、报告|implemented_pending_results|old_acc、direct before、P4 identity、逐类、floor、逐receiver|本轮是Stage2-B，不输出seen_new/H|
|JG-M09|项目.md 10.3.1|输出极轻型资源证据|runner、报告|implemented_pending_results|6400参数、epoch/step、时延、峰值显存、持久状态、forward数|与MRIOR/DADDA同row资源比较|
|JG-M10|AGENTS.md|完整loss trace、同row结果和运行报告|runner、报告|implemented_pending_results|每row 5个epoch记录；25行同row表|不得拼接单项极值|
|JG-M11|AGENTS.md|本地验证后Git提交、再同步N607|Git、报告、SYNC映射|local_verified_pending_commit_sync|`py_compile`、59项focused pytest、dry-run、commit、远端SHA|保留现有未认领改动|

## 声明边界

- 本轮只回答Stage2-B旧类target-domain适应的严格配对问题，不代替Stage2-C新类注册实验。
- `same View`定义为完全相同的三个`leo_*_weak`物理场景、历史K10 support/query物理ID顺序、support种子`seed+1000+s`、query种子`seed+2000+s`和单Query View推理；`JG_R8_LR020`固有的三场景support-only增强是方法机制，必须单列一次性enrollment前向与状态开销，不能隐藏为免费View。
- 旧125矩阵仅用作support/query ID和历史对照结果来源；当前JG predictor必须走新建的LEO_weak-only密封包和现行运行时隔离链。
