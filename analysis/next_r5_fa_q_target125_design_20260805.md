# NEXT-R5 FA-RDCE3→qKNN Target125冻结设计

状态：`DESIGN_FROZEN / IMPLEMENTATION_PENDING / NOT_RUN_AUTHORIZED`

## 1.用户覆盖与目的

用户在2026-08-05明确要求运行`FA-RDCE3+qKNN`的完整125实验，因此覆盖此前只运行Target5的资源目标。科学协议仍为`p2_min_v1`；received-IQ、physical IDs、receiver/TX、scenario、support/query split和query权限不变，不重验VALIDATED_ONCE数据。

本实验只识别FA域适应主效应，不包含CER、RPPF、D92-Lite新头或参数搜索。历史D62、D91、D92、SVRN只在完全同键的独立评分结果中比较。

## 2.一次性方法锁

- Phase1：用既有checkpoint对应的6个old类source-only聚合统计构建一个Target专用FA-RDCE3 rank-3 INT8资产；Target125五个receiver均未进入Phase1。资产不得读取Target support/query或truth。
- K5/K10：使用同一冻结闭式公式，唯一扩展为`FA_FIT_K={5,10}`。后验精度仍为`D_F+C·K·D_v^-1`；K10只增加合法独立support，不改rank、`rho=sqrt(3)`、Wiener系数、量化或目标函数。公式闭合不构成性能保证。
- K1：`FA_STRICT_BYPASS`。`DA1_REG0≡DA0_REG0`、`DA1_REG1≡DA0_REG1`，逐logit、prediction、state和资源严格alias；不得执行`a=0+固定RDCE`冒充适应。
- K5/K10的FA只由REG0 6个old类support拟合一次，`DA1_REG1`逐bit复用`DA1_REG0`状态；new support不拟合DA。
- R0为sealed checkpoint的160维非负unit`z_id`；R1严格使用`FA-RDCE3→signed unit`结果直接进入qKNN，不做ReLU或二次归一化。

## 3.完整125矩阵

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
5×5×5=125 outer
每outer×3 leo_*_weak scene×4状态
=375 scene row / 1500 logical state surface
```

K1共有25 outer×3 scene×2 REG=150个unique qKNN prediction，并产生150个DA alias；K5/K10共有100 outer×3 scene×4状态=1200个unique prediction。总计1350个unique prediction和150个alias，逻辑surface仍为1500。

四状态：`DA0_REG0=域适应前/新类注册前`、`DA1_REG0=域适应后/新类注册前`、`DA0_REG1=域适应前/新类注册后`、`DA1_REG1=域适应后/新类注册后`。REG0的seen-new/H严格为`N/A`。

## 4.必要因果与协议闭合

1.每个scene四状态共享同一old query物理ID根；两个REG1状态共享同一new query根；support/query physical ID互斥。
2.K5/K10 support必须按同一封存pool做K前缀；不得为FA另选support。K1旁路不产生FA state。
3.query逐样本、全注册类竞争；query零fit、零update、零selection；禁止truth、role、quota、true batch count和global reassignment。
4.prediction完整封存后才打开truth。不得按receiver、seed、K/new、scene或中间性能停止或补跑。
5.K5/K10各自完整报告DA主效应；K1只报告严格旁路一致性，不把0差值写成FA收益。

## 5.实现复用边界

复用D108/D92 Target125的input binder、sealed package materializer、checkpoint、8-shard拓扑、truth-open与独立scorer。新增一个160维R0 materializer、Target专用FA资产构建/绑定、两臂`DA0/DA1`×两注册phase投影、qKNN core和薄CLI。不得复制数据、重建通用平台或修改D108历史方法。

## 6.发布门

只保留：实际Git入口；query权限负测；K1 alias、K5/K10 fit/reuse和1500/1350/150计数测试；真实checkpoint no-truth smoke；独立`P0=0/P1=0`；不可覆盖run ID；Git commit；N607 preflight和资源检查。满足后立即交给唯一Luna/max runner。性能低不是健康停止条件。
