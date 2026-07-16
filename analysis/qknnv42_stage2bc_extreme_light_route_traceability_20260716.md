# qKNNv4.2正式Stage2-B/C极轻路线追踪

状态：`ACTIVE_ROUTE_EXPLORATION_LOCAL_PROTOCOL_REPAIR_REQUIRED`。

本工作严格使用`ADV3B02_CORE90_SOFT_E200`作为基座。Stage2-B旧类域适应和Stage2-C真实seen-new注册同等重要；正式结果必须同时给出注册前和注册后。Phase2的support、query及所有适配/注册/评估信号必须在Phase2边界前已叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`，clean样本和任何clean派生信号必须物理不可达。

## 当前证据边界

- JG_R8_LR020的25行K10 Stage2-B矩阵只有旧类，old accuracy均值78.8222%，不能代替Stage2-C。
- JG_R8_LR020单development Stage2-C在new5/10/20下注册后old为57.78%/52.78%/50.83%，seen-new为61.00%/37.33%/20.67%，遗忘19.44–26.39pp，属于合法负证据。
- CSIL、MoPC-HR、Orthogonal Incremental三种对照的完整严格矩阵均未达到目标；matched Stage2-C MRIOR仍缺失。
- 当前JG锁硬编码单receiver、单seed和K10；OS级访问账本、资源capsule及正式Pareto闭环尚未完成。

## 追踪表

|ID|要求|状态|证据/下一步|
|---|---|---|---|
|R01|Phase2`LEO_weak-only`且clean/clean-derived物理不可达|partial|SOMP-H screen已在feature tensor加载前拒绝clean共存cache；正式JG enrollment/apply仍需固定argv隔离和后验访问账本|
|R02|逐样本全部注册类、无角色/真实批次数/quota/global assignment|implemented|现有JG逻辑与SOMP-H批次不变测试；正式OS闭环待集成|
|R03|不可变prediction与independent scorer隔离|partial|独立before/after FP16 stage capsule已绑定method/row/input、feature、head payload和opaque query token摘要；仍需接入`.cvspred`与独立scorer|
|R04|Stage2-B注册前与Stage2-C注册后同row|pending|正式runner/scorer尚未接入新head|
|R05|真实嵌套5/10/20 seen-new TX覆盖|pending|等待5receiver逐TX×K5/K10覆盖审计|
|R06|K10开发锁定、K5独立matched确认|implemented_not_integrated|global method lock与row manifest已分离；正式项目另要求K1/K20遗忘压力|
|R07|多receiver、多seed、多场景确认|schema_implemented_data_pending|development=`20-1/713101/K10`；confirmation=`713102–713106`，713106密封包待建|
|R08|K10 old>=92%、旧类floor>=88%|pending|真实LEO矩阵未运行|
|R09|K10 seen-new 5/10/20>=92/90/86%|pending|真实LEO矩阵未运行|
|R10|K5较K10下降<=3pp|pending|K5正式链未扩展|
|R11|注册后旧类遗忘控制|pending|SOMP-H专门针对prototype拥挤，真实性能待验证|
|R12|adapter<=50k、<=20epoch、无dense query图|implemented_not_integrated|纯SOMP-H为0参数/0epoch/0step；26类三场景state=76320B、head MAC/query=13142；相关测试41项PASS|
|R13|identity-only及三种方法Pareto|pending|baseline改为独立artifact；ProtoNet 0参数/0step只能做零维不劣＋性能/MAC/状态/时延/显存Pareto|
|R14|完整日志或闭式求解诊断|pending|SOMP-H为闭式support-only，需保存几何/原型/资源诊断|
|R15|合法TX/receiver/support-query清单|pending|legacy 20-new feature NPZ因clean共存定性`PROTOCOL_INVALID_FOR_PHASE2`；等待target-only coverage audit和sealed package manifest|
|R16|自动化报告和Git提交|in_progress|根目录报告已建立；本次route prototype待提交|
|R17|每3个turn回顾目标和对话|implemented|当前计数`1/3`|

## SOMP-H首条路线

文件：`paper_reproduction/cvs_aligned/support_only_multiprototype_head.py`。

机制：全注册类support-only对角类内残差白化、每类最多2个压缩原型、centroid混合和support几何hubness惩罚。所有query使用同一score函数并独立计算；API不接收query标签、角色、配额或query集合图。持久状态和per-query MAC直接由实际tensor shape重算。部署状态可封装为无pickle FP16 capsule，使用精确成员集合和schema；任何额外query真值成员都会被加载器拒绝。

验证：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m py_compile paper_reproduction/cvs_aligned/support_only_multiprototype_head.py tests/test_support_only_multiprototype_head.py
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' -m pytest -p no:cacheprovider -q tests/test_support_only_multiprototype_head.py tests/test_jg020_stage2c_isolation.py tests/test_adv3b02_ci_heads.py
git diff --check -- paper_reproduction/cvs_aligned/support_only_multiprototype_head.py tests/test_support_only_multiprototype_head.py
```

结果：核心与协议相关测试合计41项PASS；只有既有TorchScript弃用/trace警告。`screen_support_only_multiprototype_head.py`会在feature tensor加载前拒绝任何clean共存cache；现有legacy 20-new feature NPZ已被该边界排除。SOMP-H已采用独立before/after单stream capsule、纯密封ADV3B02 z_id160、K20 prefix、support feature/head payload、opaque token摘要、注册pair与K-family绑定及FP16 flight-state推理。该结果仅证明机制、序列化和fail-closed接口可用，不是Stage2-B/C性能成功，也不授予N607正式启动权限。
