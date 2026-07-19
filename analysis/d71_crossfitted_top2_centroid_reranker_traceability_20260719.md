# D71交叉拟合top-2局部中心重排追踪

## 问题与非重复性

D62当前聚合最强，但K10/new5的after-old、min-after-old、seen-new分别只有82.22%、53.33%、84.67%，主要下尾集中在low/rain的old4、old5、new1、new3。D67–D70证明连续stacking、逐行标定、生命周期旧行交换及其support门都不能形成无交换改善。D49否证全局cosine占比，D56–D58否证全局/逐类score校准，D59–D64否证协方差位置扫描和全pair局部协方差锦标赛。

D71不重写D62的全局joint head，也不替换任何单类行。它只在D62已经给出的top-2内部，使用低方差最近中心双类判别器决定是否交换第一、第二名；无法引入第三类。该机制针对局部碰撞，不改变其余类别排序。

## 唯一机制

对before与final分别独立处理。K>=2时，以physical rank奇偶形成两个互斥、exact-once的support-held fold。每折仅在train support上拟合D62和每个匿名类别pair的最近中心方向：

```text
d_ij = mu_i - mu_j
b_ij = -0.5 * (||mu_i||^2 - ||mu_j||^2)
pair_predict_i(x) iff x dot d_ij + b_ij >= 0
```

每个pair先要求在两折held的该pair真实样本上，两类正确数都不低于D62对这两类的受限判决，且至少一类严格提高。把全部初选pair应用到held的D62 top-2后，再要求当前全部注册类TP逐类不降、FP逐类不增；否则所有pair原子回退。full support只拟合接受pair的方向。

query先执行D62全类打分。若其top-2的无序pair被接受，则仅用对应双类方向决定两者次序；不接受则精确D62。最终返回一个全注册类分数向量，query逐样本独立argmax。pair方向使用对称int8+FP16 scale/bias；matched FP32仅作量化对照。K1精确D62。

## 协议与停止边界

- 类公式和pair公式对标签置换等变；无class ID名单、old/new角色、scene、receiver、query truth、quota、真实batch类数、global reassignment或query-dependent adaptation。
- 只用固定LEO_weak support及其固定288D视图；outer-held/query不参与fit、pair选择或阈值。ground输入为0；D22 manifest仍无正式资格。
- 最终状态为D62 int8 head加稀疏int8 pair方向；状态必须低于256KiB。query只增加一次top-2和至多一个288D pair dot，不保留dense query graph。
- 相对D62必须保持aggregate、3场景、11类floor、H、forgetting、joint和混淆无交换，并至少改善A、F、J或任一floor；否则停止D71，不扫描pair阈值、权重、温度、kNN K值或场景/角色门。
- 首轮只跑receiver`20-1`、seed`713101`、K10/new5、3场景×5fold的105行。未过development gate不跑第二seed或125。

## 实现状态

独立core、锁定D62 probe和两个测试文件已实现。D71专项12/12、D42–D71全链357/357通过；全链用时82.8s。当前证据只证明公式、状态、协议和回归闭包，尚无outer性能。提交后必须在干净worktree复验，再按automation report登记的唯一命令运行真实105行。
