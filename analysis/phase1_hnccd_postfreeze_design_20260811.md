# P1-HNCCD后冻结连续几何评价合同与实现追踪

## 范围、身份与不可声明事项

本卡冻结训练根`phase1_hnccd12_20260811_v1`的后冻结接口，后冻结运行ID和输出根固定为`phase1_hnccd_postfreeze_20260811_v1`。候选仅为`F1C_HNCCD12`至`F6G_HNCCD12`；专属schema固定为`cvs.phase1.hnccd_receipt.v1`、`cvs.phase1.hnccd_lv_export.v1`、`cvs.phase1.hnccd_leo_binding.v1`和`cvs.phase1.hnccd_postfreeze_pair.v1`。稳定clean NPZ叶名`icmt_clean_l_v_proxy_final_only.npz`只是不可改的公平核文件叶，不是候选身份。manifest、LEO binding、pair JSON和receipt不得持久化旧候选identity或原始source receiver token。

本轮只验证Phase1后冻结连续几何：它不是unknown、FAR、Phase2、Phase3、注册授权、真实在轨或性能晋级结论。训练日志中的指标不是本轮结果，不能替代sealed clean、三scene LEO、fixed400 proxy或same-fold pair的原始artifact。真实12臂checkpoint、ManySig字节、sealed42步输出和独立审查仍由唯一Runner重开；本地合成检查不能解除该依赖。

## 冻结科学合同

每个clean导出必须重开当前`final_ssdg.pth`并严格验证其`hnccd_receipt`。receipt必须由`cvsrffi.phase1_hnccd.validate_hnccd_terminal_receipt`通过，且本卡只核验HNCCD实际已有字段：`B=128`、`d=160`、local4、7个source receiver的计数和SHA、固定`28`分母、共同same-physical clean/单LEO行序、三scene、strict model keys、共同`L_base→feat_joint→exact head→tx_logits`路径、新AdamW、AMP及C/G终态。C必须`enabled=false,lambda=0`并保持辅助N/A/0；G必须`enabled=true,lambda_hnccd=.02`、三个scene均有positive cell、每scene一次raw-unscaled VJP，其中LEO`feat_joint`、shared encoder和exact head weight均finite/nonzero，clean`feat_joint`与head bias均None-or-zero。每共同batch资源观察保持一对一，只能作资源记录，不能选模。

后冻结几何只从source-L clean`feat_joint`拟合。先用float64 totalized-L2：正范数行映射为`z/||z||_2`，精确零行保持零；feature、范数、geometry、NLL、unknown量或聚合量任何nonfinite即fail-closed。每类方差固定`ddof=1`，使用`0.9×class+0.1×class-equal pooled`收缩和`1e-6`下限；评分为完整对角Gaussian-NLL，连续unknown量为稳定`u=log(4)-logsumexp(-NLL)`。V和proxy绝不进入fit；L、V、proxy的全部封存行均保留为评分证据。

LEO导出只能读取既有source IQ，固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`各一次，不生成第二个观测，不做TTA、补偿门、额外forward或选择。binding必须封存same-physical TX/RX/day/class/order、场景、ManySig SHA和当前checkpoint/receipt SHA。fixed400 proxy固定days=`2021_03_01,2021_03_08`、RX=`1-1,1-19,14-7,18-2,19-2,2-1`、seed=`7281148`、每TX上限400、总数400；JSON、CSV、NPZ与physical key/current SHA必须闭合。proxy只评分，不拟合、不调参、不反馈训练或候选选择。

固定42步为`12 clean export+12 LEO export/binding+12 fixed400 proxy+6 same-fold pair`。每fold的clean四floor、LEO三scene四floor、三scene等权overall、fixed400 proxy AUROC和`mean(u_proxy)-mean(u_V)`严格正均为非补偿门；矩阵还要求clean6/6、LEO18/18、fold三sceneoverall6/6和global18-celloverall。任一门失败永久`REJECT_P1_HNCCD_PERMANENT`，不得调参、挑fold、重命名、拼接或补偿。F6必须重开F1--F5的raw clean NPZ、LEO NPZ、LEO binding、proxy JSON/CSV和当前C/G checkpoint，重新计算摘要、floor、overall、proxy gate和C/G共同receipt；不得接纳prior pair自报摘要或缓存。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|HNCCD-PF-01|身份冻结|训练根、后冻结ID、12臂、四schema和旧identity/raw-token拒绝|全部4文件|verified|三模块导入；clean/LEO/pair旧identity与raw-token负例拒绝|不改变稳定NPZ叶名|
|HNCCD-PF-02|真实receipt重开|checkpoint args、HNCCD terminal receipt、C辅助N/A/0、G三scene positive/raw VJP、资源观察|clean、LEO、pair|verified|合成C/G terminal receipt经当前HNCCD validator重开|不伪造不存在字段；真实checkpoint仍deferred|
|HNCCD-PF-03|同fold公平|共同warm-start、strict keys、新AdamW、AMP、source split、TX/class、physical/order、scene与live head路径|pair|verified|合成same-fold C/G common-binding闭合|G-only辅助字段不强求相等|
|HNCCD-PF-04|clean几何|仅clean L拟合；float64 totalized-L2零行保留、nonfinite拒绝、full Gaussian-NLL|clean、pair|verified|float64正行/零行、nonfinite拒绝和L-only合成Gaussian/NLL|V/proxy零fit|
|HNCCD-PF-05|LEO绑定|既有IQ单LEO三scene、same-physical TX/RX/day/class/order、ManySig/current SHA|LEO、pair|implemented|冻结binding路径已编译、导入并进行raw-token拒绝|真实ManySig/LEO原件由Runner重开|
|HNCCD-PF-06|fixed400 proxy|固定days/RX/seed/400；NPZ、JSON、CSV、physical key和current SHA重算|clean、pair|implemented|固定常量与raw-logit重算路径已编译、导入|真实JSON/CSV/NPZ仍deferred；只评分|
|HNCCD-PF-07|非补偿门|clean6/6、LEO18/18四floor、fold三sceneoverall6/6、global18-cell、双strict proxy门|pair|implemented|签字公平核gate映射已编译、导入|任一失败永久拒绝|
|HNCCD-PF-08|F6原件重开|F1--F5 raw clean/LEO/binding/proxy/checkpoint和current SHA重算|pair|implemented|F6 raw reload/current receipt/proxy重算路径已编译、导入|禁止prior self-report|
|HNCCD-PF-09|本地技术验证|官方`ssr-gpu`串行编译、导入、纯函数、合成receipt/binding、旧identity/raw-token对抗和CLI help|全部4文件|verified|三脚本py_compile、三模块导入、几何/receipt/负例、三份help均通过|不读性能或N607|
|HNCCD-PF-10|真实42步接口|12个真实checkpoint、ManySig、sealed42输出、独立P0/P1和性能解释|外部唯一Runner|deferred|本卡不访问真实artifact|最高剩余风险|

## 交付与状态边界

本卡只交付4个本地后冻结接口：clean sealed export、LEO export/binding、same-fold evaluator和本追踪卡。它不新增训练forward、补偿门、模型状态、缓存、第二LEO view、query访问、proxy拟合或性能报告。即使所有本地技术检查通过，状态也只能是`LOCAL_VERIFIED / NO_PERFORMANCE_INTERPRETATION`；真实artifact闭合前，HNCCD-PF-10保持`deferred`。

当前追踪计数：`verified=5,implemented=4,deferred=1,rejected=0,blocked=0`。已实际执行的本地验证为：

```text
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m py_compile code/export_phase1_hnccd_features.py code/export_phase1_hnccd_leo_features.py code/evaluate_phase1_hnccd_postfreeze_pair.py
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <导入三模块与公开API一致性检查>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <float64 totalized-L2正行/零行、nonfinite fail-closed、source-L synthetic Gaussian/NLL>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <合成HNCCD C/G terminal receipt、same-fold common binding、旧identity/raw receiver token对抗>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <三个CLI --help>
git diff --check
```

这些检查只证明本地接口可导入、receipt与公平合同的合成路径可失败闭合；没有读取真实checkpoint、ManySig、训练日志指标、sealed42步artifact或性能数值。最高剩余风险仍是首次真实12臂receipt与sealed42步原件闭合。
