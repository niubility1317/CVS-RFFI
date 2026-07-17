# D32训练期内生安全cap轻量适应实验

## 登记

- 实验ID：`d32_inloop_safe_cap_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`COMPLETE_NEGATIVE`；D32机制证伪，不晋级、不形成正式性能声明。
- 节奏：D32是D27-D29回顾后的第3轮；本轮完成后必须在D33前执行并记录新回顾。
- 目标：修复D31训练面与部署面不一致。每个Stage2-C forward从注册旧类support计算每个新类的安全非正bias，并在完全相同的带bias分数面上训练和部署；同时继续优化新类floor与旧类遗忘。
- 比较：Z0、B3诊断、C0、D32-A/B/C；6候选×3场景×5折=90行。

## 机制

旧类Stage2-B固定为B3辅助主导拼接几何和15步compact diagonal。Stage2-C冻结旧对角阵与旧权重，仅更新new suffix：

`b_j(U)=min(0,min_i(s_old(i,y_i)-s_new(i,j)-delta))`，其中`i`只遍历注册前预测正确的旧support。

|候选|Stage2-C|步数|总步数|锁定参数|
|---|---|---:|---:|---|
|D32-A|in-loop safe cap+old/new group-balanced CE|10|25|lr 0.03、delta 1e-4、anchor 0.02|
|D32-B|A+top20%新类CVaR|10|25|lr 0.03、delta 0.10、CVaR 0.35、anchor 0.02|
|D32-C|B+有限bias恢复到-4|15|30|lr 0.025、recovery 0.15、anchor 0.03|

每步重新计算cap并回滚不安全更新；最终仅按support上的新类floor、新类总体、bias接近0、较早step选择checkpoint。K=1执行质心+cap零更新旁路。最多7个新类分块，参数峰值≤2,016；无dense query图。

## 协议

- receiver `20-1`、seed `713101`、K10、5个新类、3个LEO_weak场景；沿用已验证密封support入口。
- 每个physical support只有一个已叠加LEO_weak观测；z160/FFT96/RF32只从该固定IQ确定性提取，不增加view或信道overlay。
- query为测试集且本轮不打开；无query标签、角色Oracle、真实batch类别数、类别配额或全局分配。
- clean/source不可达；Phase1 int8组件只读不可更新。当前仍是pre-formal support-only screen，不能作正式性能声明。

## 本地验证

- 新增D32 core、共享runner的candidate lock v10/fold/full/selection/receipt/CLI闭环、launcher及测试。
- D32、D31、runner、DALI、D30 envelope、D26 compact相邻测试72/72通过；`py_compile`和`git diff --check`通过。
- 随机压力覆盖2/5/10/20新类、K=1/5、A/B/C共72个状态：旧参数与旧分数前缀位级不变，bias≤0，参数≤2,016，总步数≤30，训练/部署分数面一致。
- 本地源SHA：runner `7a041be0...cb156`；D32 core `a421f914...d6dcb`。diag不修改、不上传，只核验远端`14ec9193...1ca`。

## N607计划

- 2026-07-18 07:03 CST直接preflight通过：host `dell-DSS8440`，8×RTX 3090空闲，live inventory无训练进程；检查后本地SSH/TCP22连接为0。
- 本地Git提交：`b184411c feat(stage2): add D32 in-loop safe cap route`。
- 仅同步runner、D32 core和launcher；远端SHA分别为`7a041be0...cb156`、`a421f914...d6dcb`、`1bde1442...c2efa`。远端编译、launcher语法、唯一输出不存在和继承D31/diag SHA检查通过；连接退出后SSH/TCP22为0。
- preflight与live inventory通过后，只同步runner、D32 core和launcher；其他依赖仅校验SHA。
- 远端cwd `/home/szu2070436088/2510044040/CV-SincNet`；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 命令：`D32_GPU=0 bash code/scripts/launch_d32_inloop_safe_cap_20260718.sh`。
- output：`runs/d32_inloop_safe_cap_20260718/output/support_screen_v1`；log：`logs/d32_inloop_safe_cap_20260718/support_screen_v1.log`。
- 实际启动：2026-07-18约07:05 CST，GPU0，PID `3739951`；启动后本地SSH/TCP22连接为0。landed不等于artifact完成或性能达标。

## 结果

### 执行与artifact闭环

- V1在完成计算后因共享聚合器缺少历史兼容字段`old_score_columns_bitwise_unchanged`退出，未形成artifact，未据此作性能解释。修复仅恢复raw旧分数前缀别名，并继续独立报告DALI后的旧列；定向测试28/28通过。
- V2远端命令仍为`D32_GPU=0 bash code/scripts/launch_d32_inloop_safe_cap_20260718.sh`，实际output为`runs/d32_inloop_safe_cap_20260718/output/support_screen_v2`，log为`logs/d32_inloop_safe_cap_20260718/support_screen_v2.log`，PID `3742544`，约25.05秒完成。
- 90/90折完整；6候选×3场景×5折。D32-A/B各405条Stage2-C trace，D32-C 480条，共1,290条fold trace；无NaN/Inf、OOM、Killed、Traceback或异常终止。
- V2源SHA：runner `f49f3257b52122f695748241249aee75b51bd916b853e8ecd7f98d74474a93be`；D32 core `a421f914c777b2e5afd03f3431e55945d874b3fec7d9879604887cf83bed6dcb`。远端固定diag SHA保持`14ec9193...1ca`。
- artifact SHA：training `2d20ace759f041ee2640de0554793f1a1b7dfb4cfed91203ce1c90eb7ff36109`；support `e82eda0ee869b43c134451af7fa5d4470926454652475395fc408f4040a0a987`；selection `4444932b58958495f11b4e6a52cc65faf365b91752f6ca575edf889aa4238f32`；resource `81226b2095ae8b19a2409c4b66ff7e6f0acb31105234f815051218cb512d8320`；geometry `cd0aa129437f35f49da98ce06c2893da8ea37d609a97d2b67990ff7c1db7d2a2`；receipt `4b5896524ee8f3172c37f9f0babe25117b3537e914a35c3367b845ac262f67ad`。
- selection重算与artifact一致：`selected_candidate_id=D25-C0-DIM-CONCAT`，`selected_positive_route=false`。D32-A/B/C均未通过逐类安全门。

### 90折support-held联合结果

下表是5-fold support-held结果，不是query或正式独立确认结果；所有数字保持同一候选同行。

|候选|old-before|old-after|seen-new|mean H|forgetting|旧类池化floor|新类池化floor|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11|48.33|52.67|48.97|22.78|13.33|3.33|负基线|
|B3诊断|86.67|73.33|73.33|72.65|13.33|60.00|40.00|性能上界诊断，尚非可部署正路线|
|C0|71.67|50.56|54.00|50.35|21.11|13.33|3.33|回退候选|
|D32-A|85.56|68.33|63.33|64.32|17.22|50.00|23.33|失败|
|D32-B|85.56|65.56|60.67|61.86|20.00|33.33|36.67|失败|
|D32-C|85.56|63.89|70.67|65.79|21.67|33.33|46.67|失败|

所有候选逐fold最差旧/新类floor仍为0，远低于目标old≥92%、旧类floor≥88%、5类new≥92%。B3在old-after、seen-new、H和forgetting四项同时支配全部D32候选。

### 逐场景、逐类与floor

|候选/场景|old-before|old-after|new|H|forgetting|旧/新池化floor|
|---|---:|---:|---:|---:|---:|---:|
|A-clear|88.33|76.67|68.00|71.50|11.67|60/30|
|A-low|78.33|61.67|66.00|63.28|16.67|50/10|
|A-rain|90.00|66.67|56.00|58.17|23.33|40/30|
|B-clear|88.33|68.33|58.00|61.81|20.00|20/30|
|B-low|78.33|63.33|62.00|62.41|15.00|50/30|
|B-rain|90.00|65.00|62.00|61.36|25.00|30/40|
|C-clear|88.33|66.67|76.00|70.73|21.67|20/40|
|C-low|78.33|58.33|72.00|63.80|20.00|40/40|
|C-rain|90.00|66.67|64.00|62.83|23.33|40/40|

|候选|旧类14-10/14-7/20-15/20-19/6-15/8-20|新类09f8/1c2a/b8fb/d3af/f608|
|---|---|---|
|D32-A|70.00/50.00/86.67/56.67/56.67/90.00|23.33/80.00/73.33/80.00/60.00|
|D32-B|73.33/33.33/86.67/50.00/60.00/90.00|50.00/56.67/80.00/80.00/36.67|
|D32-C|73.33/33.33/83.33/53.33/50.00/90.00|56.67/80.00/83.33/86.67/46.67|

主要旧类floor崩点是14-7，B/C仅33.33%。C的bias recovery把09f8从B3的40.00%提高到56.67%，却使f608仅46.67%并加剧旧类遗忘，证明统一释放bias不是类间联合改进。

### 完整训练日志诊断

- Stage2-B三候选一致：fold均值loss `1.02394→0.12399`，15步，每折下降。
- Stage2-C：A `0.98871→0.86167`；B `1.54031→1.22028`；C `1.79690→1.33262`，每折末值均不高于初值。
- cap后fit support old acc/floor在全部1,290条Stage2-C记录中始终100%/100%，但held old只有63.89%–68.33%；support重代入安全不能代表泛化遗忘保护。
- bias均值：A `-9.113→-9.009`；B `-9.213→-8.857`；C `-9.213→-8.533`。C释放约0.68 logit后仍需约-8.5强抑制，暴露old/new跨组score尺度或范数失配。
- A选中step10×12、9×1、5×1、0×1；B选中10×10、9×2、8×2、5×1；C选中15×13、12×1、9×1。`rollback_count=0`且无rollback事件；这是support checkpoint回选，不能称为安全回滚触发。
- DALI在每候选45折中仅1折改变结果，只减少1个old→wrong-old；新类和跨组混淆不变。D32-C的180个held old中old→new为58，wrong-old仅7；主要矛盾是新旧跨组标尺，而非old-old身份重排。

### 资源与部署审计

|候选|峰值活动参数|总步数|adaptation MAC|query head MAC|actual状态|slim投影|batch1 CPU mean/p95|
|---|---:|---:|---:|---:|---:|---:|---|
|D32-A|2,016|25|15,897,600|4,416|52,092B|16,361B|0.350–0.357/0.358–0.411ms|
|D32-B|2,016|25|15,897,600|4,416|52,086B|16,355B|0.350–0.357/0.358–0.411ms|
|D32-C|2,016|30|21,124,800|4,416|52,096B|16,365B|0.258–0.265/0.292–0.296ms|

- query head相对identity单qKNN的17,600 MAC为25.09%，减少74.91%；每query另有87个row-local标量操作，无dense query图，CUDA峰值0B。
- 延迟和MAC不含backbone及FFT96/RF32提取，不能作为端到端星上延迟声明。
- actual状态比identity FP16的35,200B约高48%；仅说明低于256KB cap。16.35KB slim是未重封的runtime投影，`current_formal_bundle_rebuilt_as_slim_medoid=false`，不能冒充已经部署落地。
- DALI当前占37,167B、授权int8组件25,428B，却只修正约1/180个旧held样本；下一轮关闭为主，仅在真正重封约1.4KB medoid后作为消融。

### support与协议审计

- receiver `20-1`、seed `713101`、K10；3场景各60 old+50 new=110，总330。before-registration为180个old，after为同一180个old+150个new，旧support精确复用。
- 每物理样本view=1、overlay=1；330/330唯一overlay token；额外IQ、overlay、derived row均为0。z160/FFT96/RF32只是同一已接收IQ的确定性operator，不增加K。三场景physical ID、parent IQ hash、overlay token两两零重叠。
- query打开0行、0标签；clean/source/cache/control-flow均不可达；角色、真实batch类数、类别配额、global assignment均为false；预测为逐样本all-registered argmax。
- int8组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，本轮是`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`。support协议证据一致，但这些结果不是正式query或独立确认矩阵。

## D30-D32三轮强制回顾

已在D33前重新阅读目标和`项目.md`，重建并搜索项目conversation index，复查D30-D32同run注册前/后指标、完整训练日志、协议/资源/selection证据。conversation index未索引到本轮尚在进行的D30-D32细节，因此以当前本地自动化报告和完整artifact为主证据。

|轮次|核心机制|old-after|new|H|forgetting|结论|
|---|---|---:|---:|---:|---:|---|
|D30|静态max-envelope+DALI|66.67|71.33|68.19|18.89|envelope 45/45旁路，DALI无可见联合收益|
|D31-B|事后bias+new CVaR|67.78|72.00|69.06|17.78|最佳D31，但训练raw面与约-7部署bias面不一致|
|D32-C|内生cap+CVaR+bias recovery|63.89|70.67|65.79|21.67|修复同面一致性，但约-9 cap仍不泛化|
|B3诊断|同IQ拼接几何+对角适应|73.33|73.33|72.65|13.33|当前联合上界，仍未过正式目标|

回顾结论：

1. 三轮均同时报告注册前old、注册后old/new、H、逐类floor和forgetting，域适应与新类注册仍为同等优先；LEO_weak-only、无clean/source、query未开、无角色/配额Oracle边界未变化。
2. 停止静态envelope、统一bias/cap、统一CVaR、DALI权重的继续扫描。D32不是超参数不足，而是old/new跨组score尺度失配和support重代入过拟合。
3. D33优先研发球面归一化new suffix+共享温度/robust radius的同尺度竞争，以classwise LOO/shot-rank选择逐类半径和预算，分别处理09f8与f608；旧类加入robust radius margin，但禁止依赖query。
4. 同轮设置B3的≤30步加速复现支线，比较Adam、L-BFGS或近闭式对角更新；目标是在保留B3联合性能的同时压低适配步数/MAC。活动参数继续≤2,016，无dense query图，不增加新数据准备。
5. D33在实现和本地压力测试后才能启动；若无法超过B3同行联合指标且改善14-7、09f8、f608 floor，则继续判负，不进入正式5 receiver×5 seed×3场景确认矩阵。
