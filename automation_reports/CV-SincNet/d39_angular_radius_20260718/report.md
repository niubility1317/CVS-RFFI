# D39全类angular-radius标准化实验报告

## 1.实验身份与当前状态

- 实验ID：`d39_angular_radius_20260718`
- 时间：2026-07-18（Asia/Hong_Kong）
- 操作者：Codex`/root`
- 当前状态：`LOCAL_IMPLEMENTATION_VERIFIED_REAL_SCREEN_PENDING`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景；只使用已验证D18固定received-IQ support，query保持sealed。
- 目标：在不改变D38-B训练轨迹、int8原型身份和30step预算的前提下，用同一类无关公式校准全部old/new类的角距离尺度，同时改善旧→新侵入和new-new通用floor。

本报告预注册D39机制和停止门。设计、实现、单测、资源通过或support screen启动均不是性能成功。

## 2.D38证据与可证伪假设

D38真实screen完成90/90行和1200条finite trace，五项artifact SHA与receipt一致，query未打开。D38-B注册前old=87.22%，接近exact strong B3的87.78%；注册后old=0.56%、seen-new=78.67%、H=0.99%，179/180个held旧样本被新类击败。new-held的true-new对top-old margin全部为正，均值6.645、最小1.204；new-new仍有32/150条错序。int8与matched FP32在outer-held的argmax变化为0。

这些结果否定了“旧头或量化导致坍塌”的解释。D38的新类方向在多数新类上有效，但原始cosine score没有表示类内离散度，new权重对旧样本过度自信。D39检验如下单一假设：用全部注册类共享的收缩radius公式把cosine改写为angular Gaussian score，可同时恢复old/new尺度可比性并改变new-new排序。

共享new-group bias或temperature不进入矩阵。对任意正温度和共享偏置，new类内部argmax不变，因此它无法直接修复32/150条new-new错序；将bias与radius同时加入会破坏单机制归因。

## 3.锁定机制

### 3.1D38-B基座

D39完全复用D38-B：288D特征为`normalize([norm(z160);4*norm([FFT96;RF32])])`；Stage2-B用old support执行20步full-batch AdamW，量化旧头；Stage2-C冻结共享metric与decoded int8旧头，用全部old+new support执行10步new-weight-only class-balanced CE、worst-class surrogate和centroid anchor，最后独立量化新头并append。

D39不改变D38 loss、学习率、optimizer、epoch、step或权重轨迹。它只在量化权重后增加0步闭式radius状态和新的统一scorer。

### 3.2角距离与旧域prior

对量化并按存储inverse norm解码后的类权重`w_c`，以及D38共享metric变换后的单位特征`h(x_i)`，定义：

\[
\theta_{ic}=\arccos\{\operatorname{clip}[h(x_i)^\top w_c,-1,1]\}.
\]

Stage2-B结束后，只用old support及注册前int8旧头计算每个旧类的二阶角离散度：

\[
m_{2,c}=\frac{1}{K}\sum_{i:y_i=c}\theta_{ic}^{2},\qquad
r_0=\max\left(\sqrt{\frac{1}{|Y_{old}|}\sum_{c\in Y_{old}}m_{2,c}},0.05\right).
\]

`r0`在新类注册前量化为FP16并冻结。new support不得重估`r0`。

### 3.3统一收缩radius

全部类使用同一公式，固定`nu=4`：

\[
r_c^2=\frac{\nu r_0^2+(K-1)m_{2,c}}{\nu+K-1}.
\]

K1时`K-1=0`，所有类严格退化为`r_c=r_0`；不构造self-OOF，也不以单样本零残差制造虚假高置信度。K≥2时，类内离散度按相同公式进入radius，未使用handle、难类名单或类专属超参数。

旧类radius在Stage2-B后量化为FP16并冻结。Stage2-C结束后，只用final int8新权重和对应new support计算新类`m2/radius`并append；不得重算旧radius、`r0`、旧权重或`log_diag`。

### 3.4统一angular Gaussian score

推理只使用量化后FP16 radius，固定`epsilon=0.001`：

\[
S_c(x)=-\frac{1}{2}\left[\frac{\theta_c(x)}{r_c+\epsilon}\right]^2-\log(r_c+\epsilon).
\]

所有注册类进入同一argmax。每个query独立评分，不构造dense query graph，不读取query角色、真实batch类数、quota或其他query；`-log(r_c+epsilon)`惩罚宽类，避免通过放大radius无条件吞噬其他类。

## 4.状态生命周期与精度ablation

正式`D39AngularRadiusState`仅保存：

- D38正式int8 base state：FP32共享`log_diag`、old/new两级int8 code、FP16 block scale/inverse norm、类注册表；
- `radius_fp16[C]`；
- `r0_fp16`和D39 schema元数据。

注册后旧D38 int8 prefix、旧`radius_fp16`和`r0_fp16`必须逐bit不变。正式state不保存FP32 target prototype或FP32 radius回退。

D39 FP32 ablation复用与正式路线完全相同的FP16 radius和`r0`，只把D38 base weight identity替换为同轨迹FP32权重。这样outer-held argmax差异只反映prototype精度，不混入radius重估差异。FP32 ablation不可晋级。

## 5.最小development矩阵

固定6候选×3场景×5fold=`90`行，每折8-shot fit、2-shot held：

|候选|角色|
|---|---|
|identity-only single-qKNN|回退与遗忘基线|
|ProtoNet CDA|独立matched基线；保留equivalence audit|
|exact legacy strong B3 FP32|最强合法target-support-only比较器|
|D38-B residual-int8|179/180侵入的结构性负对照|
|D39 angular-radius int8|唯一promotable路线|
|D39 angular-radius FP32|matched精度ablation|

direct ADV3B02仍作为相同old-held行的0-support旁路锚，不面对未注册新类，不进入90行候选数。

## 6.严格晋级门

D39 int8只有全部满足以下条件才可锁定development query：

1. D39与D38-B的注册前预测逐row完全一致；每个scene×fold×old-class不弱于exact strong B3。
2. 旧→新侵入不高于strong B3的33/180，且严格少于D38-B的179/180。
3. after-old、forgetting和全部旧类结果逐row逐类不弱于strong B3。
4. seen-new总体不低于D38-B的78.67%；new-new错序严格少于32/150。
5. 最低新类准确率和最低new-new margin均改善；选择器只计算全部新类通用最小值，不指定handle。
6. H与joint floor逐matched row不弱于strong B3，15fold聚合值严格提高。
7. D39 int8相对D39 FP32的outer-held argmax变化为0。
8. 全部radius/`r0`有限、正值、FP16可表示；旧base prefix、旧radius prefix和`r0`逐bit不变。
9. `old_radius_new_support_row_count=0`、`held_radius_fit_row_count=0`、`query_rows_used=0`；所有类共享`nu=4`与`epsilon=0.001`。
10. 正式资源≤80k trainable params、≤30epoch、≤50step、≤256KB；需单独报告radius状态字节、radius拟合标量操作、每query的`acos/log`标量操作、平均/P95时延和峰值显存。

任一关键门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，回退identity，不打开query、不进入K1/K5/K20、125 screen或确认矩阵，也不在本轮增加bias、offset或第二种radius公式。

## 7.实现与运行计划

|项目|计划|
|---|---|
|core|新增`code/cvsrffi/stage2_d39_angular_radius.py`；D38仅增加公开cosine/temperature接缝，不复制私有decode/transform|
|Runner|在`run_d25_support_only_concat.py`增加`d39_v1`、精确90行矩阵、pairwise诊断、selector、full-K10和artifact闭环|
|测试|公式golden、K1退化、old radius append-only、formal state无FP32、row-local、标签置换、int8/FP32、资源和90行integration|
|Git|本地`ssr-gpu`验证后只提交D39授权文件和共享Runner最小差异|
|N607|只有本地K10 support-held硬门通过后才preflight、最小SCP并运行；D39负结果不在N607重复|

当前goal保持active。D39 development screen不是独立确认，完整目标仍要求5receivers×至少5seeds×3scenes×K1/5/10/20×new2/5/10/20及全部性能、floor和资源门。

## 8.本地实现、审查与版本前验证

### 8.1实现范围

本地实现没有增加第二个机制或超参数扫描。D39核心复用D38-B的20+10步轨迹，通过D38公开的`before_stage2c_hook`在20条Stage2-B trace闭合后、Stage2-C开始前实际物化old`m2/r0/radius`；新类radius仍在final int8新头生成后append。formal state强制`base_state.arm=B`，并保存D38 residual-int8 base、FP16 radius、FP16`r0`和schema元数据。

Runner增加`d39_v1`固定六候选、candidate lock v17、90行矩阵、真实radius来源token SHA、跨候选held physical-token SHA、显式D39-int8/FP32预测与radius/r0/trace匹配、严格selector、selected-only full-K10状态/资源审计和D39专属artifact schema。只有D39-int8可晋级；任一门失败回退identity。

### 8.2独立审查闭环

第一次独立只读审查发现4项问题：old radius实际物化晚于Stage2-C、selector未消费显式D39-FP32候选、formal state未强制B-arm、矩阵未验证matched held物理身份。四项均在实现层修正，没有修改预注册公式或降低晋级门。第二次独立只读复审结果为`阻断=0`、`中等问题=0`，并确认full-K10 gate未使用同support拟合性能替代outer-held资格。

### 8.3验证命令与结果

```powershell
python -m py_compile code\cvsrffi\stage2_d38_strong_b3_quantized.py code\cvsrffi\stage2_d39_angular_radius.py code\scripts\run_d25_support_only_concat.py
python -m pytest -q tests\test_stage2_d38_strong_b3_quantized.py tests\test_stage2_d39_angular_radius.py tests\test_run_d39_angular_radius_integration.py
python -m pytest -q tests\test_run_d38_strong_b3_quantized_integration.py tests\test_run_d37_b3_preserving_int8_integration.py
git diff --check -- code/cvsrffi/stage2_d38_strong_b3_quantized.py code/scripts/run_d25_support_only_concat.py tests/test_stage2_d38_strong_b3_quantized.py
```

结果为D39相关53/53通过，D38/D37共享Runner回归17/17通过，总计70/70；`py_compile`与`git diff --check`通过。测试覆盖K1/5/10/20、new2/5/10/20、公式golden、old lifecycle、B-arm拒绝、row-local推理、真实radius来源、90行候选身份、9类selector反例、3类full gate反例和selected-only full-K10。

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d38_strong_b3_quantized.py`|`89ca681356e13de62414bde7681280c5e63b4267e027f6df3ee2a762775309bd`|
|`code/cvsrffi/stage2_d39_angular_radius.py`|`78018594de21ebdcb75822d4d14164ab4bbd4e41b231fdb71c0155854dbcd86c`|
|`code/scripts/run_d25_support_only_concat.py`|`51f08dc7e7ac95dcd3dd8813c4da54147a19ee2854dfcc2ae6758f523af68e22`|
|`tests/test_stage2_d38_strong_b3_quantized.py`|`de141ea3e899182904f5eee58cee8db81c78007f8e1d5f10cd11e8f38fe1d958`|
|`tests/test_stage2_d39_angular_radius.py`|`5b1dc9f98d4c5cd5a122b30d9f2d52a9a229845ebccdabf8f9075e9b3c6e1559`|
|`tests/test_run_d39_angular_radius_integration.py`|`4bc117736b84bf77cea25ecad28e53c2543054da411182f0b64009210b48244b`|

`E:\type10-7`根目录不是Git仓库；本报告的Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，根目录同名报告仅作非版本化运行镜像。此处只证明技术实现与fail-closed审计通过，不证明D39性能可晋级。真实K10/new5 support-only 90行仍未运行，query保持sealed，N607未访问。
