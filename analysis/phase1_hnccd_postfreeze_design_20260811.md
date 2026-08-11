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
|HNCCD-PF-05|LEO绑定|既有IQ单LEO三scene、same-physical TX/RX/day/class/order、ManySig/current SHA|LEO、pair|verified|12/12 LEO NPZ与binding经当前validator重开，三scene和physical/current SHA闭合|真实ManySig/LEO原件已由唯一Runner重开|
|HNCCD-PF-06|fixed400 proxy|固定days/RX/seed/400；NPZ、JSON、CSV、physical key和current SHA重算|clean、pair|verified|12/12 proxy JSON/CSV与当前NPZ raw logits重算通过|只评分；未反馈训练或停止|
|HNCCD-PF-07|非补偿门|clean6/6、LEO18/18四floor、fold三sceneoverall6/6、global18-cell、双strict proxy门|pair|verified|42步闭合；主控从6个pair原始同row数值独立重算全部门|HNCCD专属fold/global overall阈值固定为预注册`−2pp`|
|HNCCD-PF-08|F6原件重开|F1--F5 raw clean/LEO/binding/proxy/checkpoint和current SHA重算|pair|verified|F6标记raw reopen，5/5 prior binding均`raw_artifacts_recomputed=true`|禁止prior self-report|
|HNCCD-PF-09|本地技术验证|官方`ssr-gpu`串行编译、导入、纯函数、合成receipt/binding、旧identity/raw-token对抗和CLI help|全部4文件|verified|三脚本py_compile、三模块导入、几何/receipt/负例、三份help均通过|不读性能或N607|
|HNCCD-PF-10|真实42步接口|12个真实checkpoint、ManySig、sealed42输出、独立P0/P1和性能解释|外部唯一Runner+主控|verified|Runner完成技术闭合；主控在bundle SHA绑定后完成同row性能解释|最终为永久拒绝，不进入晋级|

## 交付与状态边界

本卡交付clean sealed export、LEO export/binding、same-fold evaluator和本追踪卡。它不新增训练forward、补偿门、模型状态、缓存、第二LEO view、query访问或proxy拟合。真实42步现已闭合并由主控完成性能解释；这不扩大HNCCD的训练权限或科学声明边界。

当前追踪计数：`verified=10,implemented=0,deferred=0,rejected=0,blocked=0`。已实际执行的本地验证为：

```text
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; python -m py_compile code/export_phase1_hnccd_features.py code/export_phase1_hnccd_leo_features.py code/evaluate_phase1_hnccd_postfreeze_pair.py
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <导入三模块与公开API一致性检查>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <float64 totalized-L2正行/零行、nonfinite fail-closed、source-L synthetic Gaussian/NLL>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <合成HNCCD C/G terminal receipt、same-fold common binding、旧identity/raw receiver token对抗>
. F:\App\miniconda3\shell\condabin\conda-hook.ps1; conda activate ssr-gpu; <三个CLI --help>
git diff --check
```

本地检查证明接口、receipt、公平合同与fail-closed路径闭合；唯一Runner随后完成真实12 checkpoint、ManySig、sealed42和F6原件重开，主控才读取6个pair JSON中的同row性能数值。

## 真实42步与最终边界

真实矩阵技术工件为12 clean、12 LEO/binding、12 fixed400 proxy和6 pair，技术异常为0。主控独立重算得到clean四floor`5/6`、LEO四floor`6/18`、fold三sceneoverall`6/6`、global18-cell overall`−0.451900pp`通过、proxy双strict门`1/6`。因此多个互不补偿的冻结门失败，唯一结论为`REJECT_P1_HNCCD_PERMANENT`。

运行时pair JSON沿用了ICMT旧核对fold/global overall的`≥0pp`布尔判定，而HNCCD预注册合同是`≥−2pp`。原始同row数值、clean/LEO/proxy门和最终拒绝均不受影响；HNCCD专属包装层已改为显式`−2pp`并加入边界回归，未修改旧ICMT或远端不可变原件。该修订只纠正报告语义，不能把失败门补偿为通过。

本结论只拒绝P1-HNCCD这一机制。它不构成真实unknown、FAR、注册授权、Phase2、Phase3或多卫星协同能力结论，也不得通过调参、挑fold、换checkpoint、改名或与旧机制拼接复活。
