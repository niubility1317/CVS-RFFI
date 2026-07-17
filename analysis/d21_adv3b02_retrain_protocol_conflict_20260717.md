# D21 ADV3B02重训协议冲突与决策选项

## 已确认事实

旧`ADV3B02_CORE90_SOFT_E200`的不可变checkpoint receipt记录：

```text
L=8,400
U=58,800
V=16,800
L/U/V相对source pool=0.10/0.70/0.20
rho_label=L/(L+U)=0.125
```

因此旧checkpoint不满足当前`项目.md`第5节的`rho_label<=0.1`，只能作为历史性能/codec诊断，不得重新封装为当前正式Phase1 deployment bundle。

`项目.md`当前还有两个彼此冲突的要求：

1. 第5节定义`rho_label=|L_s|/(|L_s|+|U_s|)<=0.1`。
2. 第5.1节固定`L/U/V=0.1/0.6/0.3`，其实际`rho_label=1/7≈0.142857`。

此外，第5.1节要求选择200epoch中source validation TX accuracy最高的epoch，而Git承载面的当前trainer强制`checkpoint_selection=final_only`。两项不能同时作为正式启动依据。

## 三个互斥选项

|选项|split|checkpoint selection|优点|代价/边界|
|---|---|---|---|---|
|A：公式与现有5.1叙事优先|`0.07/0.63/0.30`，`rho=0.1`|200epoch中source-val TX最佳|严格满足`rho`公式，保留30%互斥source-val和文档选择规则|需给当前Git trainer增加仅source-val可见的best选择；正式推荐|
|B：当前trainer治理优先|`0.08/0.72/0.20`，`rho=0.1`|final-only|与近期Phase1合法训练族和当前代码一致，修改量较小|必须先修订`项目.md`5.1的split与selection口径；不是旧B02等价复现|
|C：旧B02历史诊断|`0.10/0.70/0.20`，`rho=0.125`|历史joint-safe|无需重训，已有v1数据与checkpoint|只能诊断，不能formal export、seal或Stage2声明|

## 推荐

正式路线推荐A：它以`项目.md`显式数学准入和source-val-only选模为最高优先级，修复代码而不为迁就当前实现改写科学定义。新lineage应命名为`ADV3B02_RHO10_SRCVAL_E200_S392002`，明确不是旧checkpoint的等价复现。

若用户更重视与近期Phase1 final-only治理一致，则选B；必须先同步修改根目录与Git承载面的`项目.md`，再修改terminal profile并启动。无论A或B，均应：

- `from_scratch=true`，不得从旧B02初始化，否则新权重已继承超当前标注预算的信息；
- 固定seed392002、200epoch、source-only、三种LEO源训练视图，不接触target receiver/support/query；
- 训练阶段关闭旧dense/v1 prototype正式导出；
- checkpoint冻结后再以两遍流式exporter计算center、R3偏移和P90 radius；
- 新v2组件与runtime、class binding和parity receipt完成外层联合seal后，才进入正式Stage2。

## 启动门

在用户选择A或B前，不启动正式ADV3B02重训或Phase1 export。该等待不阻断D21 lifecycle、联合bundle代码和本地对抗测试继续推进。
