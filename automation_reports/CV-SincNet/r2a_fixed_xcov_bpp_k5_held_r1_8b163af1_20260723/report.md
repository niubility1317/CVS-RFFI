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
|release-control Git HEAD|`PENDING`|
|route/GPU/PID/exit|`PENDING`|
|remote ZIP/wrapper/source/input SHA|`PENDING`|
|prediction|`0`|
|score metric rows|`0/72`|
|同row四臂性能表|`PENDING`|
|最终裁决|`PREREGISTERED / NO_PREDICTION`|
