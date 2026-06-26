# CVS项目协议

## 场景定义

CVS的主场景是天基RFFI中的弱标注跨接收机域泛化与在轨跨域少样本适应。模型在地面训练，部署到目标卫星接收机域后只允许推理、prototype更新、轻量校准、阈值微调或小adapter更新。

项目需要处理四个约束：

- 星上算力受限，完整训练放在地面完成。
- 发射机身份标签稀缺，receiver、day、rx_day和信道场景等domain label更容易获得。
- 星地链路中的residual Doppler/CFO、相位噪声、低SNR、低仰角、弱多径和弱Rician/shadowed-Rician fading会破坏raw IQ中的发射机细节。
- 在轨部署会遇到旧类和新类少样本，并需要对未知类拒识。

## 集合定义

```text
x = R_d( H_d * T_y(s) ) + n
```

- `T_y`：发射机硬件非理想性，是应保留的身份来源。
- `H_d`：传播/星地信道，是域扰动来源。
- `R_d`：接收机链路响应，是跨接收机偏移来源。
- `n`：噪声。

```text
R_s = {source training receivers}
R_t = {target receiver domain / deployment proxy domain}
intersection(R_t, R_s) = empty

Y_old = ground-training transmitter set
intersection(Y_new, Y_old) = empty
intersection(Y_unknown, union(Y_old, Y_new)) = empty
```

`R_t`可以是单接收机，也可以是多接收机deployment proxy domain。关键条件是`R_t`与`R_s`不相交，并且target-old、target-new和unknown的support/query权限都按同一个`R_t`定义。

## 地面阶段

地面训练是weak-label/semi-supervised source-domain DG，不是部署few-shot。训练数据分为：

```text
L_s = {(x_i, y_i, d_i): receiver(x_i) in R_s}
U_s = {(x_j, d_j): receiver(x_j) in R_s, y_j hidden or unavailable}
rho_label = |L_s| / (|L_s| + |U_s|) <= 0.1
```

推荐`rho_label`网格为`{0.005,0.01,0.02,0.05,0.1}`。地面阶段不得使用`R_t`的样本、统计、BN信息、阈值、prototype、adapter、伪标签、验证结果或early stopping信号。

## 表征架构

```text
raw IQ -> CV-SincNet/CVS -> z_id, z_dom
```

- `z_id`用于发射机身份分类、prototype、少样本注册和旧类校准。
- `z_dom`吸收receiver、day、rx_day、channel和satellite-style nuisance，用于域诊断、域监督、adapter gate和泄漏审计。

推荐机制包括物理先验CV-SincNet、`z_id/z_dom`解耦、domain-supervised`z_dom`、GRL/leakage probe、Mean Teacher/FreeMatch/UPS、prototype agreement、MLDG/episodic source split和source-derived satellite strong-view consistency。

## 部署阶段

在轨部署阶段面对目标接收机域`R_t`。Stage2-B/C必须记录正整数`K`、support/query划分、receiver/TX split、threshold scope和satellite/LEO target view。

推荐`K`锚点为`{1,2,5,10,15,20,50}`。`K<=20`可称few-shot/low-shot；`K>20`应称higher-shot、medium-shot或saturation point。

## 可声明与禁止声明

可以声明：

- CVS面向天基RFFI的弱标注跨接收机DG与在轨跨域few-shot适应。
- WiSig/ManySig是terrestrial proxy benchmark / ground-accessible source domain family。
- satellite stress是物理启发部署压力测试。
- Stage2-B是旧类目标域校准。
- Stage2-C是seen-new enrollment，前提是`Y_new`与`Y_old`不相交，且`R_t`与`R_s`不相交。

禁止声明：

- WiSig/ManySig是真实卫星训练集。
- satellite augmentation等价于真实在轨验证。
- source-only DG等价于few-shot learning。
- 旧类target support提升就是新类识别。
- Stage2-A/B拒识结果就是seen-new identity accuracy。
- `R_t`与`R_s`重叠后仍称部署泛化。
- 缺少target-old或target-new样本覆盖时仍声称完整Stage2-C。

## Git与Markdown同步

任何CVS项目相关改动都必须进入Git可追踪流程。改动前必须运行`git status -sb`，改动后必须检查`git diff`/`git status -sb`、完成必要验证，并提交本次意图明确的变更，除非用户明确要求不要提交。

项目相关Markdown必须随代码、配置、脚本、矩阵、prompt、报告模板或协议改动同步检查：

- 工作流、Git、协作或安全规则改动，更新`AGENTS.md`。
- CVS科学场景、数据协议、receiver/TX划分、`rho_label`、Stage2-A/B/C边界、K-shot、satellite/LEO视图、指标或声明口径改动，先更新本文件。
- README、docs、实验报告或发布说明涉及的用法、结果解释、发布范围和复现边界变化时，更新对应Markdown。
