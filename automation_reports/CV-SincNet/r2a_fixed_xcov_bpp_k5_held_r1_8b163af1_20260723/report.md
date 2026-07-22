# R2A-FIXED-XCOV-BPP-K5/v1.1最小held四臂性能run报告

## 1.身份与当前状态

- run_id:`r2a_fixed_xcov_bpp_k5_held_r1_8b163af1_20260723`
- candidate:`R2A-FIXED-XCOV-BPP-K5/v1.1`
- evaluation_scope:`PHASE1_HELD_PROXY_NON_PROMOTABLE`
- protocol:`p2_min_v1`；data:`VALIDATED_ONCE_REUSED`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- 当前状态:`PREREGISTERED / NOT_LANDED / NO_PREDICTION`
- retry:`NO`

本run是coverage通过后的首个真实性能闭环，只执行冻结K5 held proxy。它不改变received IQ、物理ID、receiver/TX集合、场景、support/query划分或protocol schema，因此不重复数据验证；不访问Phase2 target/query，不运行125，不调参。

## 2.目标、假设与最小证伪矩阵

真实coverage SHA确定held receiver=`1-1`。固定6个pseudo-new类×3个`leo_*_weak`场景，共18个matched slice；每个slice同时生成：

|臂|域适应|统一分类头|
|---|---|---|
|`M0`|关闭|关闭|
|`M_DA`|RCHM|关闭|
|`M_HEAD`|关闭|BPP|
|`M_JOINT`|RCHM|BPP|

主指标为同slice的`H_old_new`，协同量为`I_syn=H_JOINT-H_DA-H_HEAD+H_M0`。只对全部18个预注册slice作算术均值，不选择receiver、类、场景或最好row；同时保留全部72个arm-row。

性能完成后的即时裁决：

- 任一build、无标签predict、truth解封score或72-row闭包失败：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- prediction和score完整，但mean `I_syn<=0`、`M_JOINT.H<=max(M_DA.H,M_HEAD.H)`、任一组件退化，或联合臂相对两个单臂在old-after、seen-new、min-old、min-new、floor下降或forgetting增加：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 全部通过只形成Phase1 held联合正收益证据；由于scope为non-promotable，不直接构成Stage2/125或推广结论。

## 3.冻结输入与版本

|项目|冻结值|
|---|---|
|方法实现commit|`8b163af1c3f43d94ef1f546da2306b43533c5046`|
|实现独立review|`P0=0，P1=0→MERGE`|
|源码ZIP|`E:/type10-7/code/snapshots/r2a_fixed_xcov_bpp_k5_held_r1_8b163af1_20260723/source_8b163af1.zip`|
|源码ZIP SHA/大小/文件数|`ab9fc348aa65f46b6718ff0476971e92d8d087dbb64795700c8b51b7c0d2ac4d / 33,227,316B / 3,930 files`|
|held模块raw ZIP SHA|`51e5d187805ed5f58d7088431e9f99d878fd5687fbecc08cd9140e51963e2bc8`|
|wrapper SHA|`30fef0e3c4403a8154098c71522fefb7b2bb012db6c573857a544c092f776b7d`|
|r8 parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b / PASS / max_abs=0`|
|archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0 / 8400 rows`|
|manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|

coverage已真实通过：physical/observation各8400唯一，6/7/4/3覆盖，168 cells、zero=0、min=32，K1/K5/K10余量31/27/22，`feature_arrays_read=[]`，`held_fold_selected=false`。

## 4.协议与状态读写

- build只读验封后的Phase1 dual archive、manifest和metadata-only coverage receipt；Phase1 lock排除held receiver及当前pseudo-new类。
- 每类每场景按coverage SHA与physical ID确定性取前5个独立物理sample为support，其余为query；support/query物理ID不重叠。
- build分别写不可覆盖的packet、密封truth sidecar和query NPZ；query NPZ只允许`query_ids,z_id`。
- predict仅接收packet与query NPZ，逐query独立在全部已注册类上决策；不接收truth、角色、配额或全局重分配信息。
- build后、predict前先冻结truth SHA；predict不接收truth。score仅在prediction落盘并验封COMMIT后使用预测前冻结值解封truth，验证18个slice、四臂、row/query绑定及prediction=logits argmax，再输出72个同row指标。
- 参数更新、optimizer step均为0；C5保持identity，C6固定rank1；资源字段由同一packet/score row携带。

## 5.本地门与已知边界

`ssr-gpu`专项与相邻回归合计19项通过；独立复审为`P0=0，P1=0→MERGE`；源码包布局和模块SHA已核验，wrapper根/Git镜像SHA一致且`bash -n`通过。

完整archive在本地复核时由原verifier因本机与N607的Torch/CUDA live execution contract不同而fail-closed，未进入build/predict，也未产生性能结果。该环境绑定不放宽；正式run在archive原生N607环境把build→无标签predict作为首段真实artifact smoke，失败即按技术失败停止。

## 6.N607发布合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/r2a_fixed_xcov_bpp_k5_held_r1_8b163af1_20260723`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source root:`<run-root>/source_8b163af1`
- 输入archive/manifest/coverage/parity只读复用r8不可变run输出；不得复制后修改。
- runner先执行direct N607 preflight，确认新run root不存在、GPU/进程/磁盘与输入SHA；只同步源码ZIP和wrapper，核验安全单根布局、held模块SHA、`py_compile`、依赖import和`bash -n`。
- 唯一启动命令：`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 &`。不得retry、restart、远端编辑、调参或启动125。
- wrapper的EXIT trap以不可覆盖方式写`pipeline.exit`；runner据此核对自然退出码，不能用进程消失或marker缺失猜测exit。
- 预期输出：`output/packet.json`、`truth.json`、`query.npz`、`prediction.json`、`score.json`、`sha256sums.txt`、`complete.marker`，以及`logs/pipeline.log`、`pipeline.pid`、`pipeline.exit`。

## 7.完成回填

|字段|结果|
|---|---|
|release-control Git HEAD|`e0707fb90c932067517210dadc74843818e7d9e5`|
|route/GPU/PID/exit|`direct / GPU0 / 420304 / 0`|
|remote ZIP/wrapper/source/input SHA|`ZIP=ab9fc348…2ac4d；wrapper=30fef0e3…776b7d；held模块=51e5d187…e2bc8；r8 parity/archive/manifest/coverage均与第3节冻结值一致`|
|prediction|`18个slice；prediction.json SHA=eb9593769100dba20451b5aa1b7d49999a2754cdaf6c2337dd6f6e854da2e7df`|
|score metric rows|`72/72；18个slice；四臂=M0,M_DA,M_HEAD,M_JOINT；score.json SHA=bf6e5f55c8c8d33e754184b30af3fb6a36b142a173156d8ff3e9cc3b0d201222`|
|同row四臂性能表|`见第10节；18个matched slice、72个arm-row已完整分析`|
|最终裁决|`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|

## 8.runner终态与回收索引

- direct preflight通过；新run root在创建前不存在。启动前GPU0空闲，未见训练或其他runner进程；只启动一次，未重试、未重启、未改远端源码。
- remote ZIP、wrapper、安全单根布局、3,930文件、held模块SHA、r8 parity/archive/manifest/coverage SHA、Python依赖import、`py_compile`和`bash -n`均通过。`output/sha256sums.txt`对packet、truth、query、prediction和score逐项校验通过，`complete.marker`内容为`R2A_HELD_ARTIFACTS_COMPLETE`。
- 已回收至`E:/type10-7/automation_reports/CV-SincNet/r2a_fixed_xcov_bpp_k5_held_r1_8b163af1_20260723/retrieved/`：`pipeline.pid`、`pipeline.exit`、`pipeline.log`、packet、truth、query、prediction、score、sha256sums和marker；本地SHA逐项等于远端。

## 9.主agent独立artifact复核

完整读取并解析回收目录的全部10个文件后，独立复核通过：

- `pipeline.exit=0`；完整`pipeline.log`只有72-row完成计数和最终marker，无warning、traceback、NaN/Inf、OOM或Killed。
- `output/sha256sums.txt`中的packet、truth、query、prediction、score逐项等于本地文件SHA；prediction SHA=`eb9593769100dba20451b5aa1b7d49999a2754cdaf6c2337dd6f6e854da2e7df`，score SHA=`bf6e5f55c8c8d33e754184b30af3fb6a36b142a173156d8ff3e9cc3b0d201222`。
- packet SHA、truth SHA=`49691b5fe216fed2d9a1f44219bd819a79817f53a9d1723b353d656db6369c2c`和prediction COMMIT=`48305dfcc408398f073e46e8403289524dcb1679e7af271568f13ce495cc51a7`均按canonical JSON独立重算通过。
- query NPZ精确只含`query_ids,z_id`；1105个query ID全部唯一，`z_id.shape=(1105,160)`、dtype=`float32`，无label或role。
- 18个prediction row、72个score metric row、四臂顺序、row/query绑定及每个prediction与相应logits argmax均独立重算通过。

因此本run具有完整真实性能结果，不再属于`NO_PERFORMANCE_RESULT`。

## 10.同row四臂结果

下表为全部18个预注册slice的算术均值；每一列只聚合同一批matched rows，不拼接不同run极值。

|arm|old-before|old-after|old adaptation gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|I_syn|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`M0`|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|0.000000|
|`M_DA`|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|0.000000|
|`M_HEAD`|0.799448|0.779418|-0.020031|0.776255|0.752412|0.776255|0.462804|0.486391|0.776255|0.020031|0.043620|0.223745|0.000000|
|`M_JOINT`|0.799448|0.779418|-0.020031|0.776255|0.752412|0.776255|0.462804|0.486391|0.776255|0.020031|0.043620|0.223745|0.000000|

`M_JOINT-M0`的mean H为+0.004646、floor为+0.042896、min-old为+0.039403、forgetting为-0.011616，但old-after为-0.005974、seen-new/BA/min-new均为-0.005530。该表面改善完全来自BPP；`M_DA-M0`所有指标为0，`M_JOINT-M_HEAD`所有指标也为0。

### 10.1全部18个slice

|pseudo-new|scene|M0 H|M_DA H|M_HEAD H|M_JOINT H|I_syn|J old-before|J old-after|J seen-new|J BA|J floor|J min-old|J min-new|J forgetting|J old→new|J new→old|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|14-10|clear|0.9115|0.9115|0.8721|0.8721|0.0000|0.8167|0.8167|0.9355|0.8422|0.5588|0.5588|0.9355|0.0000|0.0000|0.0645|
|14-10|low-elev|0.7853|0.7853|0.7698|0.7698|0.0000|0.6398|0.6398|0.9661|0.6659|0.2264|0.2264|0.9661|0.0000|0.0342|0.0339|
|14-10|rain|0.8369|0.8369|0.8734|0.8734|0.0000|0.8264|0.7910|0.9750|0.8206|0.6032|0.6032|0.9750|0.0354|0.0675|0.0250|
|14-7|clear|0.8522|0.8522|0.8241|0.8241|0.0000|0.8585|0.8424|0.8065|0.8422|0.5588|0.5588|0.8065|0.0161|0.0225|0.1935|
|14-7|low-elev|0.3244|0.3244|0.3631|0.3631|0.0000|0.8230|0.7733|0.2373|0.6659|0.2264|0.2264|0.2373|0.0497|0.1211|0.7627|
|14-7|rain|0.7415|0.7415|0.7082|0.7082|0.0000|0.8715|0.8576|0.6032|0.8206|0.6032|0.7692|0.6032|0.0139|0.0208|0.3968|
|20-15|clear|0.9169|0.9169|0.8715|0.8715|0.0000|0.8190|0.8190|0.9310|0.8422|0.5588|0.5588|0.9310|0.0000|0.0222|0.0690|
|20-15|low-elev|0.7802|0.7802|0.7466|0.7466|0.0000|0.6851|0.6429|0.8904|0.6659|0.2264|0.2264|0.8904|0.0422|0.0422|0.1096|
|20-15|rain|0.8243|0.8243|0.8404|0.8404|0.0000|0.7930|0.7930|0.8939|0.8206|0.6032|0.6032|0.8939|0.0000|0.0105|0.1061|
|20-19|clear|0.8911|0.8911|0.8656|0.8656|0.0000|0.8698|0.8222|0.9138|0.8422|0.5588|0.5588|0.9138|0.0476|0.1460|0.0862|
|20-19|low-elev|0.3516|0.3516|0.3494|0.3494|0.0000|0.8933|0.7652|0.2264|0.6659|0.2264|0.2373|0.2264|0.1280|0.1372|0.7736|
|20-19|rain|0.4202|0.4202|0.7946|0.7946|0.0000|0.8462|0.8217|0.7692|0.8206|0.6032|0.6032|0.7692|0.0245|0.1154|0.2308|
|6-15|clear|0.9065|0.9065|0.8624|0.8624|0.0000|0.8214|0.8214|0.9077|0.8422|0.5588|0.5588|0.9077|0.0000|0.0000|0.0923|
|6-15|low-elev|0.7732|0.7732|0.7471|0.7471|0.0000|0.6452|0.6452|0.8873|0.6659|0.2264|0.2264|0.8873|0.0000|0.0000|0.1127|
|6-15|rain|0.8075|0.8075|0.8444|0.8444|0.0000|0.7917|0.7917|0.9048|0.8206|0.6032|0.6032|0.9048|0.0000|0.0104|0.0952|
|8-20|clear|0.8403|0.8403|0.6890|0.6890|0.0000|0.8984|0.8984|0.5588|0.8422|0.5588|0.8065|0.5588|0.0000|0.0033|0.4412|
|8-20|low-elev|0.7214|0.7214|0.7241|0.7241|0.0000|0.6730|0.6698|0.7879|0.6659|0.2264|0.2264|0.7879|0.0032|0.0317|0.2121|
|8-20|rain|0.7747|0.7747|0.7975|0.7975|0.0000|0.8182|0.8182|0.7778|0.8206|0.6032|0.6032|0.7778|0.0000|0.0000|0.2222|

### 10.2逐场景与逐类

|scene|M0 H|M_DA H|M_HEAD H|M_JOINT H|mean I_syn|
|---|---:|---:|---:|---:|---:|
|clear|0.886422|0.886422|0.830771|0.830771|0.000000|
|low-elev|0.622687|0.622687|0.616698|0.616698|0.000000|
|rain|0.734188|0.734188|0.809767|0.809767|0.000000|

|class|M0/M_DA逐类acc|M_HEAD/M_JOINT逐类acc|delta|
|---|---:|---:|---:|
|14-10|0.964238|0.958862|-0.005376|
|14-7|0.574709|0.548971|-0.025738|
|20-15|0.983455|0.905128|-0.078327|
|20-19|0.466630|0.636480|+0.169850|
|6-15|0.928962|0.899926|-0.029036|
|8-20|0.772711|0.708160|-0.064551|

held receiver只有`1-1`，K固定为5；support/query由coverage SHA确定性排序，没有可汇总的随机seed，也没有跨receiver或跨seed置信区间。逐类acc在该固定after-registration查询集上按18个slice平均；原始72-row完整值保存在回收`score.json`。

## 11.机制诊断与因果裁决

|比较|after预测变化|old wrong→correct/correct→wrong|new wrong→correct/correct→wrong|mean ΔH|结论|
|---|---:|---:|---:|---:|---|
|`M_DA-M0`|0/6630|0/0|0/0|0.000000|DA决策退化|
|`M_HEAD-M0`|798/6630|225/265|45/53|+0.004646|有局部重排，但救援少于伤害|
|`M_JOINT-M_HEAD`|1/6630|0/0|0/0|0.000000|唯一变化为wrong→wrong|
|`M_JOINT-M0`|798/6630|225/265|45/53|+0.004646|完全由head贡献|

RCHM在C6确实生成rank1非标量metric：minimum eigenvalue为0.899966–0.900064、condition为1.111032–1.111154、sqrt update Frobenius norm为0.051283–0.051335。它改变logits（`M_DA-M0`max abs=3.963478），但6630次after argmax零变化，正中冻结falsifier“`M!=I`但prediction不变”。

`M_JOINT`相对`M_HEAD`只有1次预测标签改变，发生在pseudo-new=`14-7`、rain场景、query=`source|14-7|1-1|2021_03_08|1|893`：truth为14-7，HEAD误判14-10，JOINT改为另一错误类20-19，未形成救援。18/18个slice的`I_syn`均精确为0，mean/min/max均为0；两个功能没有形成可验证协同。

## 12.资源、量化、coverage与缺失测量

|状态|rank|accounted wire/state bytes|support build MAC|postprocess MAC/query|optimizer steps|
|---|---:|---:|---:|---:|---:|
|C5 before|0|20,260|640|800|0|
|C6 after|1|22,087|5,936|1,126|0|

上述MAC只覆盖报告中冻结的matmul ledger，不是端到端时延。全部数值低于128KiB、0.34MMAC和8kMAC/query帽。

- support bank INT8+FP16量化：max abs error为0.001874–0.002236，mean abs error为7.947e-5–8.867e-5，reconstruction cosine min为0.999982–0.999987；BPP compiled-stat FP16 max abs error为0.001211–0.001347。
- after top1 logit margin最小值：M0=0.201096、M_DA=0.201233、M_HEAD=0.213959、M_JOINT=0.139877。该margin与support量化误差不在同一量纲，不能据此冒充float-teacher量化一致性。
- 本run没有float-teacher prediction、端到端平均/P95时延或峰值显存artifact，因此正式INT8 top1一致率、large-margin flip门、latency和VRAM均标记`NOT_MEASURED`，不得从代码测试或GPU空闲值推断。
- coverage沿用真实通过receipt：8400行，physical/observation各8400唯一，6/7/4/3，168 cells、zero=0、min=32，K5最小query余量27；未重复数据验证。

## 13.最终裁决与下一步

最终状态：`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

独立`gpt-5.6-sol high`结果审计完整重算18个slice、72个arm-row和6630次query，重算值与`score.json`最大差为`2.22e-16`，确认row/query、logits→argmax、四臂均值、逐场景/逐类及转移计数无误；裁决`P0=0，P1=0→MERGE_REPORT`。

直接证伪原因是DA组件在全部6630次after决策中完全退化，且18/18个slice的`I_syn=0`；同时联合臂不优于HEAD臂。该revision停止，不发布target窄实验、不运行125稳定性screen，也不以BPP在floor/forgetting上的局部改善替代联合协同要求。

runner期间完成的只读下一候选spike将`JOINT-CID-BPP/r0`裁为`MERGE_SPIKE`：复用同一K5 held协议，以support-only C-id替换RCHM并保留统一BPP，先解决outer-train/nested LODO锁和C-id/BPP残差重复收缩两个可证伪问题。它尚不是`DESIGN_FROZEN`，在独立监督MERGE前不改代码或发布实验。
