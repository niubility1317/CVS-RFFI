# CSIL与MoPC-HR全量ADV3B02移植失败定位与无LEO诊断追踪

日期：2026-07-23

## 结论

v7不是运行失败，而是方法移植、基础状态和任务分布共同导致的真实性能崩塌。当前代码不能表述为“官方仓库代码仅换成CVS数据，核心完全未改”。准确名称应继续使用`CSIL-on-trainable-ADV3B02 CVS adaptation`和`MoPC-HR-on-trainable-ADV3B02 CVS adaptation`。

最高风险问题是v7复用的`paper_full_base_state.pt`只有80条source样本，而当前builder在相同6旧类、7source接收机、2天、每组合100条配置下实际返回8400条，旧产物少了105倍。CSIL的Fisher/旧指纹和MoPC-HR的旧原型都由该80条状态产生。

## 证据闭环

|ID|要求/假设|实现或证据|状态|结论|
|---|---|---|---|---|
|TR-01|v7完整执行|8个分片各100cell；800cell、2400正式行；归档3300份JSON|verified|不是中途失败或日志截断|
|TR-02|CSIL官方早层冻结|官方`CSIL.m`对`learnableFpLayerIdx-3`之前的gradient mask置零|verified|与“ADV3B02全量不冻结”直接冲突|
|TR-03|当前CSIL全量解冻|`adv3b02_paper_full_ci.py`报告`backbone_frozen=false`，实际更新约38.8万至39.2万参数|verified|属于核心训练语义改变|
|TR-04|CSIL指纹扩展一致|官方按新增类别扩展feature/fingerprint坐标；当前固定增加32维随机线性投影|rejected|架构并非逐层等价|
|TR-05|CSIL小样本批处理一致|官方60/40随机划分、`floor(N/20)`丢尾；当前分层60%、每类至少1条且不丢尾|rejected|K1等小样本条件的实际更新步不同|
|TR-06|MoPC-HR优化器主参数一致|官方SGD、lr0.01、momentum0.9、wd2e-4、batch16、20epoch；当前相同|verified|这些标量未改|
|TR-07|MoPC-HR原型增强一致|官方旧原型加0.05高斯噪声并参与CE；当前相同主机制|verified|主机制保留|
|TR-08|MoPC-HR层级正则一致|官方逐parameter使用衰减系数和L2 norm；当前按semantic module聚合并使用平方L2|rejected|正则量纲与权重分布改变|
|TR-09|MoPC-HR原型纠正一致|官方raw dot-product后softmax；当前paper cosine权重|rejected|核心原型迁移权重改变|
|TR-10|MoPC-HR蒸馏损失遗漏|官方虽计算`loss_distillation`，但公开trainer最终loss未加入该项|verified|当前不加KD与公开trainer实际执行一致|
|TR-11|source基础状态规模正确|v7 receipt为80；相同当前配置只读重建计数为8400且每旧类1400|rejected|v1旧状态复用错误是高风险根因|
|TR-12|无LEO诊断只改变新类信道|新runner按`tx/rx/day/eq/sig`匹配ManyTx同物理记录，只替换`target_new`IQ；旧类IQ逐数组hash保持不变|verified|满足单因素配对设计|
|TR-13|无LEO结果可作为正式CVS结果|协议明确标为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`|rejected|只能用于归因|
|TR-14|训练loss可从v7完整回收|800个现有回执未持久化逐step loss trace|deferred|新诊断runner开始保存首/末/均值loss摘要|

状态计数：`verified=8`、`rejected=5`、`deferred=1`、`blocked=0`。

## 性能崩塌机制

### CSIL

1. 官方方法依赖冻结旧特征块和旧指纹块；全量解冻ADV3B02后，新类support上的CE会更新几乎全部骨干参数。
2. KD只在新类输入上约束旧类logit，并没有旧类样本上的决策边界监督。
3. 当前固定32维新增投影不是官方“随新增类别扩展坐标”的同构实现，新类指纹只占新增块，基础160维被置零。
4. 80条source状态使Fisher和旧指纹代表性不足。现有Fisher活动值范围约`1.0–1.019`，接近均匀，难以提供有选择性的参数保护。
5. 结果因此表现为旧类尚有部分残留，但新类学习很弱：总体old后34.76%、new5.27%、H6.22%。

### MoPC-HR

1. 每个增量阶段只用当前新类真实样本CE，旧类主要依靠带噪原型CE和HR约束。
2. 全量解冻约38.3万至38.6万参数，且每个新类阶段重新训练20epoch；new20累计4个阶段、每场景560步，灾难性遗忘被反复累积。
3. 当前平方L2、semantic-layer系数与官方逐parameter非平方L2不同，`beta=1`下不能假定与官方正则强度等价。
4. 80条source旧原型进一步削弱旧类锚点。
5. 结果表现为新类可学但旧类几乎归零：总体old后2.74%、new34.20%、H3.48%，new20所有K的old后不超过0.24%。

## 无LEO诊断锁

- v7基础状态SHA、checkpoint、方法代码和参数保持不变。
- receiver、seed、new count、K-shot、物理样本ID、support/query顺序及旧类LEO IQ保持不变。
- 只把新类support/query替换为ManyTx同一`tx/rx/day/eq/sig`物理记录的归一化未叠加IQ。
- 三个结果切片仍对应v7的三种旧类LEO条件；新类IQ在三切片相同，不能解释为三个无LEO场景。
- prediction文件先封存并计算SHA，之后才计算指标。
- 该轮只回答“新类LEO叠加造成了多少损失”，不能修正80条base-state或方法核心差异。

## 后续必须分开的实验

1. 先完成同base-state的LEO/无LEO配对诊断。
2. 再用当前builder重新生成8400条source base-state，重复LEO矩阵；这是实现修复，不是参数调优。
3. 若要求仓库逐代码等价，应另建官方语义分支：CSIL恢复早层冻结和按新类扩维；MoPC-HR恢复逐parameter非平方L2与dot-softmax原型纠正。该分支与用户要求的“ADV3B02全量不冻结”不能同时声称严格官方等价。

## 验证命令

```text
conda run --no-capture-output -n ssr-gpu python -m py_compile paper_reproduction/scripts/run_adv3b02_paper_full_newclass_no_leo_diagnostic.py
conda run --no-capture-output -n ssr-gpu python -m pytest -q tests/test_adv3b02_paper_full_newclass_no_leo_diagnostic.py
```

