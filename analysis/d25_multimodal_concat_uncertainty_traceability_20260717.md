# D25同一LEO_weak接收IQ多表征拼接与不确定度注册追溯表

日期：2026-07-17

状态：设计锁定，代码实现前

主路线：`z_id160 + FFT96 + RF32 = 288D`分块归一化拼接

## 设计结论

每个Phase2物理IQ样本只对应一条已经叠加一次且仅一次LEO_weak信道的观测。`z_id160`、`FFT96`和`RF32`只是同一条received-IQ上的三个确定性数学feature block，最终拼成一条288维特征；它们不是三个样本、不是三种LEO状态、不是多view训练，也不增加K。

现有D1使用`norm([norm(z), 4*norm([fft,rf])])`，辅助分支占`16/17=94.12%`平方能量。D25保留高维拼接，但废止该失衡权重。初始块能量按维数比例固定为

\[
(\pi_z,\pi_f,\pi_r)=(160,96,32)/288=(5/9,1/3,1/9).
\]

对每个块先独立L2归一化，再拼接

\[
\Phi(x)=\left[\sqrt{\pi_z}\,N(z),\sqrt{\pi_f}\,N(f),\sqrt{\pi_r}\,N(r)\right]\in\mathbb R^{288}.
\]

因此`cos(Phi_i,Phi_j)`严格等于三个块余弦的固定加权和。后续可在support-only开发阶段测试受限块内对角阵，但不得用query选择块权重或对角参数。`0.07/0.63/0.30`保留为既有方法的损失项权重，不作为模态权重。

旧类的Phase1 int8地面锚只存在于160维身份块。对旧类`c`，先在z块执行D24不确定度融合，FFT/RF块保持纯target：

\[
p^{old}_{c,z}=N(\lambda_c t_{c,z}+(1-\lambda_c)g_{c,z}),\quad
p^{old}_{c,f}=t_{c,f},\quad p^{old}_{c,r}=t_{c,r}.
\]

其中`lambda_c`由ground/target support半径按逆不确定度闭式确定。新类完全独立地由target support注册：

\[
p^{new}_{c,b}=t_{c,b},\quad b\in\{z,f,r\}.
\]

Stage2-C追加新类后，旧类编码payload、块半径、计数、融合权重、块权重和对角阵必须逐字节冻结。评分对所有注册类采用同一个逐样本方程，不读取query真标签、old/new角色、真实批次类别数、类别quota、query顺序或全局分配信号。

## 需求—实现追溯

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D25-01 | 用户最新决定 | 最终特征采用高维拼接，不压回160维 | `code/cvsrffi/stage2_multimodal_concat_fusion.py` | verified | `test_concat_is_288d...` | 默认块能量按维数比例 |
| D25-02 | `项目.md` Phase2 | 每个物理样本只有一个LEO_weak观测；三个块不增加K或support行 | 同上、runner receipt | verified | 资源字段、错行拒绝、operator lineage测试 | runner写`support_view_count=1`与`derived_support_rows=0` |
| D25-03 | D22失败分析 | 删除D1辅助分支94.12%能量支配 | 同上 | verified | W4与D25块能量同测 | 不删除FFT96/RF32维度 |
| D25-04 | 用户指定三层原型 | int8 ground-old仅提供身份先验；FP32 target-old做域校正；FP32 target-new纯target注册 | 同上 | verified | ground-z隔离与纯target-new测试 | ground没有FFT/RF字段 |
| D25-05 | D24设计 | 旧类z块使用ground/target不确定度融合 | 同上 | verified | D24依赖回归与ground component替换测试 | 复用D24数学语义，不改D24历史模块 |
| D25-06 | 用户防遗忘要求 | Stage2-C追加新类后旧类状态和旧类score列冻结 | 同上、tests | verified | old-prefix、payload、score bitwise测试 | 旧类不重估radius |
| D25-07 | K-shot要求 | K=1/5/10均合法；K=1半径使用预登记`r0` | 同上、tests | verified | K1 old-target/new三块半径测试 | 旧类融合z半径由闭式融合产生 |
| D25-08 | floor优化 | 记录逐块半径、类间距与`gap=d(p_i,p_j)-rho_i-rho_j` | 同上 | verified | 三种pair role与gap测试 | floor hard gate仍在runner层实现 |
| D25-09 | 对角阵探索 | 支持288维块内去均值、受限对角阵；FFT/RF不再有不同数量级边界 | 后续adapter模块 | deferred | 待基础闭式路线通过再实现 | 首轮先验证0参数拼接，避免盲目增参 |
| D25-10 | 快速梯度更新 | Stage2-B最多50正式steps/75探索steps；Stage2-C优先0-step闭式注册 | runner | pending | 完整training log与step审计 | 正式上限以`项目.md`为准 |
| D25-11 | query=test | fit/append接口不含query/truth/role/quota/global-assignment | 同上、tests | verified | public API签名与单样本全注册评分测试 | predictor只接受单样本特征 |
| D25-12 | 资源Pareto | 分列head MAC、FFT `O(TlogT)`、RF32 `O(T)`、状态、临时scratch、延迟、显存 | module audit、runner report | implemented | 模块资源schema测试 | 实测延迟/显存仍待runner |
| D25-13 | 压缩要求 | 默认保存FP32 target原型；FP16/int8仅作预登记Pareto消融 | module audit、D23 bank | deferred | D23格式bank已有独立回归 | D25先锁FP32几何，后做paired format ablation |
| D25-14 | 正式协议 | 正式路线必须共同密封checkpoint、int8组件和method lock | bundle/runner/report | blocked | 需support-only正向证据后重建 | 历史int8组件仅可用于授权screen |
| D25-15 | 实验矩阵 | support-only原子门后才进入K1/5/10和正式5RX×5seed×3scene×new5/10/20 | runner/report | blocked | 需D25基础模块与方法锁通过 | 不用平均值绕过逐类floor门 |
| D25-16 | D25 runner设计 | 独立runner复用D19密封support控制面，但不修改历史D19 runner；固定Z0/B3/C0/C1/C2共75行 | `code/scripts/run_d25_support_only_concat.py` | verified | 候选锁、B3等价、75行矩阵测试 | B3仅diagnostic，不可晋级 |
| D25-17 | 证据闭包 | support打开前锁定D25核心、D24、CIAF、D19 helper、D25 runner SHA，并输出6个无样本特征artifact | runner、tests | verified | source closure与artifact schema测试 | v1发现依赖遗漏；v2补齐五成员并切独立output |
| D25-18 | 注册顺序 | 纯target与ground旧类均严格继承manifest预注册class handle顺序，不按类名字典序重排 | D25核心、runner、tests | verified | 非字典序registry回归与runner fold检查 | v2发现；v3显式传入`registered_classes` |
| D25-19 | 旧列确定性 | 注册前后旧类score列必须bitwise相同；旧prefix与新suffix分离点积，禁止以容差放宽 | D25核心、tests | verified | 6旧+5新随机几何bitwise回归 | v3发现形状相关末位差；v4修复 |

## 首轮候选锁

1. `D25-C0-DIM-CONCAT`：固定`(5/9,1/3,1/9)`，纯target原型，0epoch。
2. `D25-C1-UF-GROUNDZ`：C0+旧类z块D24不确定度融合；FFT/RF仍为target-only，0epoch。
3. `D25-C2-BLOCK-RADIUS`：C1+逐块收缩半径评分，K1使用预登记`r0`。
4. `D25-C3-DIAG-FLOOR`：只有C2通过15/15 support-fold原子非劣门后，才启用288维受限对角阵与floor-margin目标。

不恢复辅助权重4，不把地面z锚补零或伪造为地面FFT/RF原型，不把多个候选做ensemble。每个候选对每个物理support始终只产生一条288维行。

## 首轮验收门

- 三个feature block来自同一received-IQ token/hash，行数完全一致；`support_row_multiplicity=1`。
- 逐类、逐receiver、逐场景同row比较；old floor、new floor、遗忘和`H_old_new`均为硬门。
- Stage2-C后旧类prefix hash与相同输入的旧类score列bitwise不变。
- query打开前完成候选锁；query只由隔离predictor生成不可变prediction artifact，再由独立scorer连接truth。
- adapter、epoch/step、状态、MAC、时延、显存满足正式上限；开发放宽候选不能直接进入正式矩阵。

## 2026-07-17实现验证

- 新增`code/cvsrffi/stage2_multimodal_concat_fusion.py`和`tests/test_stage2_multimodal_concat_fusion.py`。
- `ssr-gpu`解释器：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- `python -m py_compile code\cvsrffi\stage2_multimodal_concat_fusion.py tests\test_stage2_multimodal_concat_fusion.py`：PASS。
- `python -m pytest -q tests\test_stage2_multimodal_concat_fusion.py tests\test_stage2_uncertainty_proto_fusion.py tests\test_stage2_target_prototype_bank.py`：37项PASS。
- 新增`code/scripts/run_d25_support_only_concat.py`与`tests/test_run_d25_support_only_concat.py`；runner不提供query/truth/scorer/role/quota/global-assignment/source/clean CLI或public fit参数。
- `python -m pytest -q tests\test_run_d25_support_only_concat.py tests\test_stage2_multimodal_concat_fusion.py tests\test_stage2_uncertainty_proto_fusion.py tests\test_stage2_target_prototype_bank.py`：44项PASS。
- v1远端在import阶段因缺少D24文件退出，support/query均未打开；v2把D24与CIAF纳入closure，`tests/test_run_d25_support_only_concat.py`、D25、D24、CIAF共42项PASS。
- v2在合法support首fold发现纯target旧类字典序漂移并fail closed，query未打开；v3显式绑定manifest registry，runner+D25+D24+D23共45项PASS。
- v3发现6列与11列点积kernel导致旧score末位差并fail closed；v4分离计算old prefix/new suffix，runner+D25+D24+D23共46项PASS。
- 当前追溯状态：13项verified、1项implemented、2项deferred、2项blocked、1项pending。最高风险项是D25-10：受限对角阵/floor梯度目标及正式50-step预算尚未接入隔离runner；当前实现严格对应0epoch闭式拼接—注册核心，不是完整训练路线。
