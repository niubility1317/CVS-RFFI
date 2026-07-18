# D54 D46谱transport开发报告

## 1.状态与目标

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；operator Codex。
- 开发单元：receiver20-1、seed713101、K10/new5、3场景×5fold；本地运行，不访问N607、不运行125。
- 目标：保持D46 classwise LOO的new优势，同时用D53安全谱transport改善old/rain；不新增任何尺度或门控。

## 2.公式与协议

公式与D53相同：`G=U M0^T/||M0||2^2`，`DeltaW=diag(gamma)G W0`，但`W0`来自D46。K1/K2在谱检查前精确D46 fallback。复用`VALIDATED_ONCE p2_min_v1`胶囊；support-only；query/role/quota/count/global/clean/source/dense query graph禁止。

## 3.文件、验证与停止门

- `code/scripts/probe_d54_d46_spectral_contracted_median_transport.py`
- `tests/test_probe_d54_d46_spectral_contracted_median_transport.py`
- D54＋D46联合27/27、`py_compile`通过。
- 必须至少保持D46的new84.67%、min-new73.33%、after81.67%，并改善H/forget/joint/floor之一且不产生场景交换伤害；失败即停止，不扫尺度/clip/第二seed/formal/125。
- 完成后报告7候选、3场景、逐类、15fold、D45/D46/D51/D52/D53比较、20epoch、混淆、谱、量化、资源、artifact SHA，并执行D52–D54三轮回顾。

## 4.执行锁

- 实现提交`0b06631e`；clean worktree`E:\type10-7\code\snapshots\d54wt`；探针SHA`f99427ba8606c0905c7e0f82534cb40519dede4b5c44d80f1dc2ae3d007541a0`；输出启动前不存在。
- exact command与D53报告第7节逐参数相同，仅作以下3处确定替换：脚本为`probe_d54_d46_spectral_contracted_median_transport.py`；arm为`--d54-arm d46_spectral_contracted_median_transport`；输出为`E:\type10-7\automation_reports\CV-SincNet\d54_d46_spectral_transport_probe_20260719\d46_spectral_contracted_median_transport`。所有seal/envelope/manifest/binding hash、runtime、device、mode和candidate-set不变。

## 5.总体、场景与逐类性能

完成105/105行、exit0、elapsed`73.305s`、query0。7候选中D54 int8/FP32同为：before`92.22%`、after`81.11%`、new`84.00%`、H`81.40%`、forget`11.11pp`、joint`23.33%`、min-before/after/new`80.00/53.33/76.67%`，混淆`26/7/17`。其余候选与同锁D52/D53账本一致：B3`87.78/75.56/72.67%`，HNBR new`15.33%`，BEC after`20.56%`，Z0/ProtoNet`71.11/48.33/52.67%`。

|场景|before|after|new|H|forget|joint|min-after|min-new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|98.33%|90.00%|98.00%|93.57%|8.33pp|40%|70%|90%|4/1/0|
|low-elev|88.33%|78.33%|74.00%|74.36%|10.00pp|20%|60%|50%|8/4/9|
|rain|90.00%|75.00%|80.00%|76.28%|15.00pp|10%|30%|70%|14/2/8|

逐类old before→after：O0`90→90%`、O1`96.67→90%`、O2`96.67→90%`、O3`80→53.33%`、O4`100→73.33%`、O5`90→90%`；new：N0/N1/N2/N3/N4=`76.67/86.67/76.67/90/90%`。D54抬高了new floor，但O3仍是old floor，O4仍有明显遗忘。

## 6.十五个outer行

|场景|fold|before/after/new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100/100/90%|94.74%|0pp|50%|100/100/50%|0/1/0|
|clear|1|100/83.33/100%|90.91%|16.67pp|0%|100/0/100%|1/0/0|
|clear|2|91.67/83.33/100%|90.91%|8.33pp|50%|50/50/100%|1/0/0|
|clear|3|100/91.67/100%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|clear|4|100/91.67/100%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|low|0|91.67/66.67/80%|72.73%|25pp|50%|50/50/50%|4/1/1|
|low|1|66.67/58.33/70%|63.64%|8.33pp|0%|50/50/0%|1/0/3|
|low|2|91.67/91.67/50%|64.71%|0pp|0%|50/50/0%|0/2/3|
|low|3|100/100/80%|88.89%|0pp|0%|100/100/0%|0/0/2|
|low|4|91.67/75/90%|81.82%|16.67pp|50%|50/50/50%|3/1/0|
|rain|0|83.33/83.33/60%|69.77%|0pp|0%|50/50/0%|2/0/4|
|rain|1|100/58.33/90%|70.79%|41.67pp|0%|100/0/50%|5/1/0|
|rain|2|91.67/83.33/80%|81.63%|8.33pp|50%|50/50/50%|1/0/2|
|rain|3|91.67/75/90%|81.82%|16.67pp|0%|50/0/50%|3/0/1|
|rain|4|83.33/75/80%|77.42%|8.33pp|0%|50/50/0%|3/1/1|

## 7.相对性能、机制、量化与资源

|版本|after|new|H|forget|joint|min-after|min-new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D46|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|
|D53|81.67%|83.33%|81.28%|10.56pp|23.33%|53.33%|73.33%|26/8/17|
|D54|81.11%|84.00%|81.40%|11.11pp|23.33%|53.33%|76.67%|26/7/17|

相对D46改变4/15行：after`-0.56pp`、new`-0.67pp`、H`-0.93pp`、forget`+0.56pp`，min-new`+3.33pp`、new→old`-1`，但old→new`+1`、new→new`+2`。相对D53改变3行：new`+0.67pp`、H`+0.12pp`、min-new`+3.33pp`，但after`-0.56pp`、forget`+0.56pp`。没有联合超越D46。

谱机制final correction L2 min/mean/max=`0.0061/0.1248/0.7797`，transport norm/bound均值`0.2166/0.4672`，界严格通过；before为`0.0158/0.0926/0.2351`。量化argmax/margin/support变化`0/0/0`，最大score误差`0.001848`。额外适配430,272 MAC-equivalent，总适配`1,077,758,242`，query MAC6,624，参数2,016、state8,583B、CUDA22,886,912B；query/role/quota/count/global/clean/source全0/false。

## 8.Artifact

|文件|大小/B|SHA256|
|---|---:|---|
|metadata|1,894|`fe5e332ed67efce58c2ec30f19cf96871d9d70ee1d03400de0e70ae560512f3e`|
|receipt|4,940|`4cb4734f3a0c21dd0d381cfcf53400f780b8eda9695f2b26817bcc057244ed87`|
|selection|2,990|`8d75fc7eec0e04c30507346cb1f36604ce3e4c0469f3143061cb510b963eeaf5`|
|support audit|313,592|`8a9958782975fc6dc5389a5c9ee4ddf3299d4fdc47998d1b438b48d927adc4ef`|
|training log|43,546,251|`e9b8f1258dd1dd1c0344a003d29beb77b0fa3294a46daa59a31331783034e85a`|
|full summary|94,573|`f8722866be15bda5af62d44c0d4fee810d37cb09330a43b1b402758ae7846e71`|

## 9.D52–D54三轮回顾

已重新核对活动目标和`项目.md`约束；三版均在相同run同时报告before/after/new、H、遗忘和逐类floor，domain adaptation与new registration同等审查。固定`VALIDATED_ONCE p2_min_v1`数据未重验证；`leo_*_weak-only`、support-only、no-clean/source、no-query-truth/role/quota/count/global assignment全部闭合。

|版本|核心尺度|after|new|H|forget|min-after|min-new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---|
|D52|gamma×base norm|81.67%|80.00%|79.96%|8.89pp|66.67%|66.67%|修正过大，向old偏移|
|D53|D45谱transport|81.67%|83.33%|81.28%|10.56pp|53.33%|73.33%|尺度安全，收益不足|
|D54|D46谱transport|81.11%|84.00%|81.40%|11.11pp|53.33%|76.67%|new floor改善但总体退化|

成功经验：D52确认median方向包含old floor信号；D53/D54确认谱映射可把修正稳定到0.1量级；D46底座仍保持最佳new/H联合点。淘汰路线：停止全部median系数残差、base-norm缩放、谱transport及其底座替换；不扫描强度、不加clip/role/scene门控。下一轮必须回到D46，研究与centroid残差正交的新机制，优先处理rain old遗忘与low-elev new floor的共同support-margin结构，而不是继续缩放同一方向。回顾完成前及本报告提交前不启动D55。

D54最终`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；当前最强仍为D46，仍不满足项目要求，不运行125。

## 10.D55类无关LOO难度截距补偿预注册

D55回到未经残差修改的D46。对每类使用D46已经合法计算的full/block inner-LOO CE与其classwise融合权重：

```text
d_c = sum_g w_g,c * CE_g,c
Delta b_c = d_c - mean_j(d_j)
W_D55 = W_D46
b_D55 = b_D46 + Delta b
```

高LOO-CE困难类得到正截距补偿，容易类得到负补偿；中心化保证无全局常数漂移。无系数、温度、阈值、clip、扫描、class ID、old/new角色、scene/receiver或query；K1/K2精确D46 fallback。D55只运行一次同开发单元，完整性能要求不变；若new/H/floor与old/forget不能联合改善即停止。
