# D92 E0 FULL MAXMIN FLOORBOOST Hard11实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|d92_e0_full_maxmin_floorboost_hard11_20260812_v1|
|状态|LOCAL_VERIFIED / APPROVED_FOR_N607|
|日期|2026-08-12|
|操作者|Codex主Agent；N607由唯一专用runner执行|
|候选|E0_FULL_MAXMIN_FLOORBOOST|
|claim scope|DEVELOPMENT_ONLY_FLOOR_HARD_SCREEN|
|协议|p2_min_v1，复用既有VALIDATED_ONCE capsule/split|
|目标|显著提高最弱旧类floor，同时大幅降低旧类遗忘，并保持H、旧类均值和已见新类准确率|

本报告在实现和N607发布前建立。E:\type10-7根目录不是Git仓库；本报告的版本化承载面是E:\type10-7\code\snapshots\d92_125wt中的同名镜像。

## 2.事实、假设与三轮回顾

完整Target125上，E0_FULL_ONLY相对D92的总体变化为H+0.2372pp、旧类均值+0.1600pp、已见新类+0.3233pp、遗忘-0.1600pp、floor-0.2133pp；但本轮冻结Hard10上其弱点集中暴露：mean delta H=-0.8201pp、旧类均值=-1.2778pp、floor=-3.6667pp、已见新类=-0.3417pp、遗忘=+1.2778pp，且8/10行floor下降。

此前E0OCF证据表明，单独固定旧类contrast融合只能小幅提高floor，不能稳定改善H和遗忘；因此本轮不再扫描OCF权重，也不复活Fisher、LOO或160维Lite。新假设是：同注册后FULL/BLOCK的固定contrast注入，加上基于旧support低分位margin缺口的有界零和max-min bias校正，能够把容量从强旧类重新分配给弱旧类，降低旧类内部混淆和新类侵入，同时不改变旧类组均值和新类头。

2026-08-12已刷新项目conversation index并检索D92、E0_FULL_ONLY、OCF、floor和forgetting；结论与上述本地报告一致。首次检索只因本地终端GBK不能输出负号而失败，设置UTF-8后成功，不属于项目验证失败。

## 3.冻结公式

保留288维A、robust center B、task-balanced covariance C、E=OFF、F0查询头和FULL主头。K1/K2严格沿原D92 FULL alias。

K>2时，所有校准只发生在DA1_REG1，DA1_REG0不参与最终头拼接：

1. 在同一DA1_REG1 support上分别拟合一次FULL和一次BLOCK；
2. 对旧类contrast去均值，以旧support class-centered logit RMS把BLOCK对齐到FULL；
3. 以contrast_lambda=0.25固定注入BLOCK旧类contrast；
4. 在融合头上，对每个旧类计算正确类相对全部其他注册类的20%低分位support margin，并扣除同一DA1_REG1中旧类margin与全类margin之差的均值，显式惩罚新类注册竞争漂移；
5. 对retention分数经tanh、旧类组内去均值和最大绝对值归一，在旧类bias上执行上界为0.35倍FULL旧类RMS的零和max-min校正；
6. 新类weight/bias逐字节保持FULL，旧类weight均值和bias均值保持FULL；
7. 仅数值校准退化时回退E0_FULL_ONLY；registry、标签、query访问或新行不变量漂移必须报错。

冻结标量只有三个：contrast_lambda=0.25、margin_quantile=0.20、retention_bias_kappa=0.35。分位算法固定为NumPy method=lower，使K5取最小样本、K10取第二小样本；它是算法身份而非第四个可调标量。候选query结果返回前禁止修改。

## 4.冻结矩阵

Performance outer共10个：

1. rx_7_7__seed_713106__k_10__new_5
2. rx_7_7__seed_713104__k_5__new_20
3. rx_7_7__seed_713103__k_10__new_5
4. rx_8_8__seed_713103__k_5__new_20
5. rx_8_8__seed_713103__k_10__new_5
6. rx_8_8__seed_713106__k_5__new_20
7. rx_7_14__seed_713104__k_10__new_10
8. rx_3_19__seed_713102__k_10__new_5
9. rx_7_7__seed_713105__k_10__new_20
10. rx_7_7__seed_713104__k_10__new_5

K1 liveness outer为rx_20_1__seed_713106__k_1__new_20，不进入性能均值。每个outer固定leo_clear_weak、leo_low_elev_weak、leo_rain_weak，共11outer、33scene-arm、单arm、11job、8shard。

历史D92和E0_FULL_ONLY不重跑。冻结对照文件为E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv，SHA256=6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a。

## 5.晋级与改进门

相对D92的原floor/H/旧类均值/新类门保持不变：mean floor不低于0、至少8/10行floor不降、最差floor不低于-2.0pp、mean H不低于0、至少8/10行H不降、mean旧类均值和已见新类均不低于0。

旧类遗忘升级为独立硬目标：

- mean delta forgetting不高于-0.5pp；
- 至少8/10行遗忘不增加；
- 最差单行delta forgetting不高于+0.5pp；
- 相对E0_FULL_ONLY mean forgetting至少改善1.8pp；
- 进取目标为相对D92 mean forgetting不高于-1.0pp且9/10行不增加。

相对E0_FULL_ONLY还要求mean floor至少+4.0pp、mean H至少+0.8pp、旧类均值至少+1.0pp、已见新类不下降。全部最低门通过才可裁决ADVANCE_TO_FULL125；但不得自动启动full125。改进友善但未全过门时仅允许REVISE_ONCE_FLOORBOOST，且仍须相对E0_FULL_ONLY将mean遗忘至少改善1.0pp；floor提升不足2.0pp或已见新类下降超过0.5pp时REJECT_FLOORBOOST。

## 6.资源与协议门

- K>2 two-state component fit不超过4，DA1_REG1 actual component=2；
- K5相对D92 fit至少降低91.67%，K10至少降低95.45%；
- 无K-fold、无Fisher/Pareto、无cross-state head splice；
- query MAC与E0_FULL_ONLY完全一致，永久state不增加；
- hard矩阵注册wall P90不高于180ms，增量peak P90不高于3MiB；
- query truth/fit/update/selection/role/quota/global reassignment全部为false；
- 每个query独立在全部注册类上竞争，新候选不得改变行集、权重或门槛。

## 7.最小数据与发布对接

只确认11个outer来自合法Target125、support/query物理ID不变、三场景齐全、support/query不重叠、scorer在不可变prediction提交后读取truth。不得重做IQ、物理ID、receiver、TX或场景验证。

计划远端路径：

|用途|路径|
|---|---|
|source root|/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_maxmin_floorboost_source_snapshot_20260812_v1|
|output root|/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_maxmin_floorboost_hard11_20260812_v1|
|logs root|/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_maxmin_floorboost_hard11_20260812_v1|
|Python|/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|

计划唯一启动命令：

    cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_maxmin_floorboost_source_snapshot_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &

仅在本地聚焦测试、真实checkpoint truth-free K1 smoke、独立P0=0/P1=0、Git提交和普通N607 preflight通过后发布。重复数据验证、额外SHA包装、通用权限系统、P2美化和完整叙事不得阻塞。

## 8.实现与验证状态

|项目|状态|证据|
|---|---|---|
|设计与三个标量|FROZEN|本报告与实现计划|
|Hard11行集|FROZEN|本报告第4节|
|核心TDD|PASS|FloorBoost公式、单臂注册、四态收据与数值回退；所属三文件55项通过|
|机械runner/analyzer TDD|PASS|Hard11矩阵、smoke、8shard、共享停派、三方分析与裁决；11项通过|
|主线整合回归|PASS|11个相关测试文件共130项通过；包含既有E0OCF与E0_FULL_ONLY回归|
|静态检查|PASS|7个生产文件py_compile、config JSON、两个CLI help、git diff --check均通过|
|独立P0/P1|PASS|独立Terra复审：P0=0、P1=0、APPROVE|
|Git commit|PASS|方法与Hard11实现commit=`25047050`；发布控制面commit待填|
|N607 preflight/smoke/run|PENDING|待填|
|完整取回与分析|PENDING|待填|

核心实现已经明确区分四种状态：DA1_REG0不激活、K1/K2精确D92 FULL别名、K>2正常FloorBoost、K>2仅数值退化时完整FULL头逐字节回退。registry、标签、query访问和新类行漂移不进入回退分支，必须报错。尚未读取任何新候选性能。

### 8.1冻结发布包

|项目|冻结值|
|---|---|
|方法commit|`25047050`|
|runtime archive|`d92_floorboost_runtime_closure_25047050.tar.gz`；5014801B；1302成员|
|archive SHA256|`da99c0d36541c12b355f947c706092f569d04fbfb7e26b8e7c8f6773282041dc`|
|method lock SHA256|`8d5ca23cc2f0cdbe995dd7cb5ba382550d9ca172ec2d81e39abd029669bfd510`|
|launch SHA256|`c2e9afce282e2b5d041a12f1549039149f2cde5f7f541bca03f42b7e25d72c48`|
|archive闭包|包含`code/cvsrffi/__init__.py`、FloorBoost核心/runner及D81/D92复用入口；不存在`code/code`层级|

同步映射固定为：archive→`source_root/d92_floorboost_runtime_closure_25047050.tar.gz`；method lock→`source_root/configs/stage2_d92_full_maxmin_floorboost_hard11_v1.json`；launch→`source_root/launch.sh`。`launch.sh`先核对两份SHA和全新路径，再解包、核对import闭包、prepare 11job、执行真实checkpoint truth-free K1 smoke；只有smoke闭合后才启动8个shard。

## 9.结果表

实验完成后必须在此填入D92、E0_FULL_ONLY和E0_FULL_MAXMIN_FLOORBOOST三方法同排表、10行paired表、三场景分解、floor/遗忘最差行、资源表和最终裁决。运行前不填写或推测性能。
