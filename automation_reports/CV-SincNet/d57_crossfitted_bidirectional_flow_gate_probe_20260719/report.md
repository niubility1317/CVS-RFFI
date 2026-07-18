# D57交叉拟合双向混淆流门报告

## 1.状态与问题

- 状态：`PREREGISTERED_PRE_IMPLEMENTATION`；operator Codex；不运行125。
- 固定receiver20-1、seed713101、K10/new5、3场景×5fold development cell；复用`VALIDATED_ONCE p2_min_v1`。
- D56把after从D46的81.67%提高到83.33%、forget从10.56pp降到8.33pp，却把new从84.67%降到80.67%、min-new从73.33%降到60.00%。D57只修复这一可观测交换，不修改D46的B20、full/block head、RMS、classwise权重、量化或query路径。

## 2.预注册机制

对D56已经合法生成的每折support inner-held D46分数，折`r`的流修正只能由其余`K−1`折构造：

`Delta b_c^(-r)=(out_c^(-r)-in_c^(-r))/((K-1)*C)`。

对每个匿名类`c`，分别统计基础D46与“只加入坐标`Delta b_c^(-r)`”后的：

- `positive_correct_c`：真实类为`c`的held样本预测正确数；
- `false_positive_c`：真实类不为`c`但被预测为`c`的held样本数。

仅当`positive_correct_adjusted>=positive_correct_base`、`false_positive_adjusted<=false_positive_base`且至少一项严格改善时，`accept_c=1`；否则为0。然后把同一mask应用到全support D56流：`delta_c=accept_c*Delta b_c`，删除类公共常数后一次性加入D46截距。

为避免多个坐标联合后产生交互伤害，再以同一cross-fit方式同时应用所有accepted坐标；若任一类的positive correct下降或false positive增加，则`atomic_fallback=true`并精确返回D46。K1/K2也精确D46 fallback。

## 3.协议、对称性与禁止项

全部证据来自support inner-held分数和support标签；outer-held/query/clean/source不可达。公式对类标签置换等变，不使用class ID、old/new角色、scene、receiver或handle。无alpha、temperature、clip、threshold、坐标顺序、贪心迭代、第二arm或参数扫描。before/final分别按同一公式独立拟合；这不是按角色分支。最终只保存一套int8/FP16 affine state，逐query独立全类argmax，dense query graph为0。

## 4.成功门与停止门

D57必须至少保持D46的before92.22%、after81.67%、new84.67%、H82.33%、forget≤10.56pp、min-after53.33%、min-new73.33%、joint23.33%，并严格改善after/forget/floor至少一项；三场景不得交换伤害；INT8/FP32翻转为0/0/0；至少1个final prediction改变。失败即停止，不放宽门、不扫描，不跑第二seed/formal/125。

完成后必须详细报告7候选、3场景、11类、15fold、D46/D56同折变化、每类base/adjusted positive与false-positive计数、accept mask、atomic fallback率、补偿分布、20epoch、量化、资源和全部artifact SHA。D57完成后执行D55—D57三轮技术复盘。

## 5.实施计划

1. 复用D56一次额外inner-score refit，不增加第二套head或query state。
2. 添加单类双向门、联合原子门、rank/class置换、K1/K2、无坐标顺序、资源和tamper测试。
3. `ssr-gpu`窄验证、Git提交、clean detached worktree锁定后，只运行一次105行本地development矩阵。
4. 当前不访问N607；若后续候选通过开发门，远端动作须先执行规定preflight。
