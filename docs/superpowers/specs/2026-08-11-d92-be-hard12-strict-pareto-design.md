# D92-BE-Hard12严格Pareto瘦身实验设计

|字段|值|
|---|---|
|日期|2026-08-11|
|分支|`codex/d92-be-hard12-strict-pareto-20260811`|
|设计状态|用户已批准“性能提升且计算量下降”的严格Pareto方向；书面规格待用户复核|
|实现状态|未开始|
|首轮实验代号|`D92-BE-2x2-Hard12-v1`|

## 1.目标与判定原则

本轮只回答一个问题：能否在不牺牲D92旧类、新类和floor表现的前提下，删除低贡献部件并显著降低注册构造成本，同时使`H_old_new`得到可辨识提升。

候选只有在性能与计算两个维度同时过门时才能晋级。计算下降但性能持平、性能提升但计算不降、或者以旧类/新类/floor退化换取速度，均判为失败。Hard12只承担development筛选；任何正式推广结论仍需候选冻结后的完整Target125确认。

本规格冻结首轮四臂因果实验。首轮结果返回后，若没有严格Pareto候选，后续新机制必须另行冻结method lock；不得通过修改Hard12行、指标权重或成功阈值挽救失败候选。

## 2.协议与声明边界

- 协议固定为`p2_min_v1`。
- 输入仅为冻结Phase1 deployment bundle、已封存的固定LEO弱信道received IQ、当前row的合法support标签和注册表。
- 复用匹配的`VALIDATED_ONCE`、`capsule_id`和`split_id`；方法、开关、资源计量和method lock变化不得触发数据重验证。
- query逐样本面对全部已注册类，不能参与fit、update、selection、rollback、阈值或候选选择。
- 本轮不增加DA。报告使用`DA0_REG0`和`DA0_REG1`；`DA1_REG0`与`DA1_REG1`记为`N/A(method lock disables DA)`，不使用含混的“before/after”。
- `DA0_REG0`在四臂间复用同一不可变预测面；新类准确率与`H_old_new`仅在`DA0_REG1`定义。
- Hard12由历史结果选择，因此只能标记为`DEVELOPMENT_ONLY_COVERAGE_CONSTRAINED_STRESS_SCREEN`，不能称为无偏性能估计、fresh confirmation或Target125替代物。

## 3.历史证据与瘦身选择

证据来自`E:\type10-7\automation_reports\CV-SincNet\cvs_full_ablation_completed_matrix_analysis_20260731\report.md`及`E:\type10-7\github_publish\CVS-RFFI-repo\docs\D92_METHOD_COMPLETE_REPORT_20260727.md`。

|部件|历史配对证据|本轮决定|
|---|---|---|
|A：288维联合特征|`FULL−A0`的`ΔH=+0.3081`，95%CI为`[0.2803,0.3261]`|保留；禁止恢复160维Lite|
|B：地面扰动谱与Cauchy稳健中心|`FULL−B0`的`ΔH≈+0.00003`，主指标CI均跨0|首个删除因子|
|C：旧/新任务均衡协方差|`ΔH=+0.0027`、`Δold=+0.0080`、`Δold_floor=+0.0071`，但`Δnew=−0.0021`|首轮保留，避免同时改变任务权重|
|D：full/block3与LOO可靠性融合|D0/D1/D2同时绕过Fisher或rollback，历史证据不是纯D消融|首轮保留，不作D结论|
|E：Fisher残差与逐类Pareto统计选择|`ΔH=+0.0014`、`Δold=+0.0044`、`Δold_floor=+0.0051`、`Δnew=−0.0010`；只接受34/5175个类别场景单位|主要计算删除因子，但需验证B×E交互|
|F：头部存储格式|F3压缩约49%，但batch1头部延迟`0.1157ms`高于F0的`0.0190ms`|固定F0；本轮不把存储压缩写成计算加速|

历史Stage2-C仅覆盖3个development seed和1个new-class draw。上述结果只能确定首轮因果优先级，不能预判`B0E0`一定成功。

## 4.四臂方法锁

四臂共享以下不可变路径：

- `A=joint288`：160维identity、96维FFT和32维RF组成同一288维联合特征；
- `C=task_balanced_covariance_0.5_0.5`；
- `D=full_block3_loo_reliability_fusion`；
- 通用finite检查、数值失败fallback、rollback和原子状态提交保持开启；
- `F=f0_fp32_weight_fp32_bias`；
- 同一Phase1 bundle、同一support/query、同一等先验全注册类argmax；
- `K<=2`沿原精确fallback路径，不读取其性能作为瘦身收益。

|实验臂|B|E|实现语义|角色|
|---|---:|---:|---|---|
|`FULL`|开|开|原D92注册路径|配对参考|
|`B0`|关|开|`support_plain_mean_no_ground_spectrum`，其余与FULL相同|B主效应|
|`E0`|开|关|关闭Fisher residual及其统计Pareto选择，保留通用工程安全闭合|E主效应|
|`B0E0`|关|关|同时采用B0和E0，其余与FULL相同|唯一主要晋级候选|

`B0`、`E0`只解释主效应，不单独晋级。四臂同时存在才能估计B×E交互，避免把历史未运行的组合效应假设为零。

## 5.注册计算假设

对`K>2`，FULL和B0的闭式分量拟合数为

```text
N_fit(FULL/B0)=8(K+1)
```

关闭E后只保留基础full/block主拟合与LOO：

```text
N_fit(E0/B0E0)=4+4K=4(K+1)
```

因此：

|K|FULL/B0|E0/B0E0|理论下降|
|---:|---:|---:|---:|
|5|48|24|50%|
|10|88|44|50%|

K10现有保守资源清单中的Fisher稠密代数上界为8.409GMAC。删除E可以删除该审计项，但不能把8.409GMAC直接写成实测时延。实现必须重新产生每个arm、每个outer的`N_fit`、注册wall time、process CPU time、peak working set和输出状态字节receipt。

最终query仍是同形状的单一仿射头。注册后26类的分类下界保持`26×288=7,488MAC/query`；本轮不得增加query分支、support检索或query侧Fisher计算。

## 6.Hard12-v1冻结矩阵

### 6.1输入身份

|输入|SHA256|用途|
|---|---|---|
|D92 retry2 `row_metrics.csv`|`bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62`|D92五指标历史困难度|
|NEXT-R5 r11 `score.json`|`fa2344ae037e4ab5dfec6fea9bb0f534c7d5c9cdeb3596797bdc403b3c9fcc23`|独立路线四指标历史困难度|
|D92/D131 `target125_context.json`|`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`|同一密封D92 package链和物理数据绑定|

D92与NEXT-R5的125个outer key精确匹配。NEXT-R5的准备上下文SHA等于D131保存上下文，该上下文逐row绑定原始D92 retry2的before/after enrollment/apply package根。

### 6.2困难度公式

对125个outer采用平均秩处理并列。升序平均秩记为`rank_avg∈[1,125]`：

```text
pi_down(x)=(125-rank_avg(x))/124
pi_up(x)=(rank_avg(x)-1)/124
```

低值更难的指标使用`pi_down`，高遗忘更难使用`pi_up`：

```text
D92_i=mean[
  pi_down(H_old_new),
  pi_down(c_old_acc),
  pi_down(c_old_floor),
  pi_down(seen_new_acc),
  pi_up(average_forgetting)
]

R5_i=mean[
  pi_down(H_old_new),
  pi_down(old_balanced_accuracy),
  pi_down(old_floor),
  pi_down(seen_new_acc)
]

Hard_i=0.5*D92_i+0.5*R5_i
```

其中R5的四个输入指标先在同一outer的三个LEO场景、`DA0_REG1`状态下等权平均，再进入125个outer的秩变换。

二进制MILP最大化所选`Hard_i`之和，约束如下：

- outer总数为12；
- 每个receiver出现2至3次；
- 每个seed出现2至3次；
- `(K=1,new=20)`恰好2次；
- 其余每个`(K,new_count)`切片出现2至3次；
- 强制包含全局共识最难sentinel `rx_3_19__seed_713104__k_1__new_20`；
- 同目标值按`(receiver,seed,K,new_count)`字典序和`1e-9`固定扰动闭合。

本规格对应的canonical manifest草案SHA256为`26ca470a4cc79d13498493863e6958c3fc5c82af1b3dbecd06cf6277d0a650e4`。实施时生成的manifest若不能复现该摘要，必须在发布前停止并解释差异。

### 6.3精确outer清单

每个outer固定运行`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`三个互斥场景。

|角色|outer key|Hard|
|---|---|---:|
|K1 liveness|`rx_20_1__seed_713103__k_1__new_20`|0.922278225806|
|K1 liveness|`rx_3_19__seed_713104__k_1__new_20`|0.962600806452|
|performance|`rx_20_1__seed_713105__k_10__new_20`|0.574294354839|
|performance|`rx_20_1__seed_713106__k_5__new_20`|0.726108870968|
|performance|`rx_3_19__seed_713103__k_10__new_10`|0.632358870968|
|performance|`rx_3_19__seed_713106__k_10__new_20`|0.778830645161|
|performance|`rx_7_14__seed_713102__k_10__new_20`|0.509979838710|
|performance|`rx_7_14__seed_713102__k_5__new_20`|0.660584677419|
|performance|`rx_7_7__seed_713104__k_10__new_5`|0.168447580645|
|performance|`rx_7_7__seed_713105__k_10__new_5`|0.190725806452|
|performance|`rx_8_8__seed_713104__k_10__new_10`|0.423991935484|
|performance|`rx_8_8__seed_713105__k_5__new_20`|0.635786290323|

覆盖闭合为：receiver计数`{20-1:3,3-19:3,7-14:2,7-7:2,8-8:2}`；seed计数`{713102:2,713103:2,713104:3,713105:3,713106:2}`；切片计数`{K1/new20:2,K5/new20:3,K10/new5:2,K10/new10:2,K10/new20:3}`。总计12outer、36scene；性能判定只使用10outer、30scene，K1的2outer、6scene只做fallback和liveness闭合。

## 7.严格Pareto成功门

所有性能差值均按`candidate−FULL`计算。统计单位是outer；先对同一outer的三个LEO场景等权平均，再在10个performance outer上汇总，禁止把30个scene伪装成30个独立样本。

`B0E0`只有同时满足以下全部条件才通过：

### 7.1性能提升

1. `mean(ΔH_old_new)>=+0.005`，即至少提高0.5个百分点；
2. 10个performance outer中至少8个满足`ΔH_old_new>=0`；
3. `mean(Δold_balanced_accuracy_at_DA0_REG1)>=0`；
4. `mean(Δseen_new_acc_at_DA0_REG1)>=0`；
5. `mean(Δold_floor_at_DA0_REG1)>=0`；
6. `mean(Δaverage_forgetting)<=0`；
7. 四臂的`DA0_REG0`预测必须逐值相同，否则视为方法锁漂移而非性能结果。

报告同时给出paired outer bootstrap 95%CI、per-receiver、per-slice、per-scene和per-class old结果，但development Hard12不以CI包装成正式确认。

### 7.2计算下降

1. K5与K10的闭式分量拟合数分别从48降至24、从88降至44；
2. 在同一N607节点、同一CPU亲和/线程设置、同一输入和计量口径下，从特征已就绪到候选状态原子提交的配对注册wall time中位数至少下降40%；
3. 同一批outer的注册器增量peak working set中位数至少下降40%；该值扣除冻结encoder、已加载特征和公共运行时的基线RSS，不能用端到端进程RSS掩盖注册器差异；
4. query仿射头维度、MAC计数和代码路径不增加；
5. 不把MAC等价上界、并发调度时长或Python单次噪声冒充硬件时延结论。

任一门失败即`NO_STRICT_PARETO_PROMOTION`。B0或E0即使单独更好，也只生成因果证据；是否形成后续候选需在下一轮重新冻结，不能在本轮运行中临时替换晋级对象。

## 8.最小发布门与错误处理

本轮只保留直接防止实验跑错的门：

1. Git-backed方法入口和冻结method lock；
2. `VALIDATED_ONCE`、`capsule_id`、`split_id`、`p2_min_v1`句柄匹配，不重复数据验证；
3. 聚焦协议负测：query zero-fit、zero-update、zero-selection，以及禁止clean/source/query-truth/role/quota/global-reassignment访问；
4. 真实checkpoint、无query truth的预测smoke；
5. 独立审查`P0=0`且`P1=0`；
6. 本地Git commit、不可覆盖run ID/output/report、一次N607预检与资源占用记录；
7. 先完成不可变prediction closure，再由独立scorer连接truth。

不增加重复数据追溯、额外签名层、通用发布平台、恶意同进程防御或与当前实验无关的P2治理。

运行中只按技术健康停止：P0协议/安全违规、错误checkout或hash、输出覆盖风险、launcher级确定性故障，或者至少两个不同row在生成prediction前出现同一异常指纹。禁止依据H、accuracy、floor或其他中间性能停止。停止时只终止已绑定到该run的进程树，保留全部部分artifact，并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 9.实现边界与测试设计

实现阶段应把B和E开关限制在同一D92共同代码路径，不复制四套方法。配置、状态receipt和method lock必须显式记录`B_enabled`、`E_enabled`、`A/C/D/F`固定值、fallback身份及资源计数。

最低测试集合：

- B0严格落到`support_plain_mean_no_ground_spectrum`；
- E0只删除Fisher residual/Pareto统计路径，不删除finite、rollback和原子提交；
- 四臂保持288维、相同C/D/F0、相同query全类argmax；
- `K<=2`四臂精确fallback/alias闭合；
- K5/K10的`N_fit`分别闭合到48/88与24/44；
- query zero-fit、zero-update、zero-selection负测；
- manifest输入哈希、Hard12行集、场景数和覆盖计数闭合；
- prediction在truth/scorer之前完成且不可变；
- resource receipt包含wall time、CPU time、peak working set、fit count、head bytes和query MAC；
- 真实checkpoint truth-free smoke；
- 独立审查最终返回`P0=0,P1=0`。

## 10.实验发布节奏

首个N607发布一次性运行冻结四臂×Hard12，得到48outer-arm、144scene-arm的完整矩阵。每个run ID只有一个runner，primary agent不重复启动。状态依次为`LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE -> ANALYZED`。

后续方法迭代继续复用同一Hard12和相同成功门，每个候选使用新的method lock、Git commit、不可覆盖run ID和报告。任何新候选结果不得反向修改Hard12。连续完成三轮探索后，按`AGENTS.md`先做记录化回顾，再决定第四轮。只有Hard12严格Pareto通过的冻结候选，才有资格进入完整Target125确认。

## 11.明确不做的事项

- 不恢复D92-Lite 160维路线；
- 不使用类别顺序、真实角色或query真值解tie；
- 不把FA-RDCE3、qKNN附加头或DA混入首轮四臂；
- 不在首轮改D、C权重、F格式或query头；
- 不把K1结果写成性能收益；
- 不用Hard12替代完整125或发布promotable性能声明；
- 不因候选或资源计量变化重复验证数据。

## 12.规格复核结论

本规格把“提升性能，计算量下降”落实为不可拆分的严格Pareto门。首轮变量仅为B与E，A/C/D/F及数据、query和安全闭合保持固定；Hard12在候选运行前由两套历史125证据和覆盖约束冻结。该范围足以回答首轮因果问题，也足够小，可以在最低必要发布门通过后直接进入真实N607实验。
