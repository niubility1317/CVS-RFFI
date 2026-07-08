# Receiver-AgnosticTwo-stageUDA多子agent复现审计报告

日期：2026-07-08

对象：`paper_reproduction/receiver_agnostic_twostage_uda`

论文：Receiver-AgnosticRadioFrequencyFingerprintingBasedonTwo-stageUnsupervisedDomainAdaptationandFine-tuning

审计性质：paper-faithful复现一致性检查。该审计不评价CVS Stage2、LEO部署、open-set/new-class注册或半监督DG扩展。

## 1.审计编组

|角色|agent id|审计范围|
|---|---|---|
|Architecture/Preprocess Auditor|`019f3fd4-e9aa-7220-902a-263e8275680b`|网络结构、输入shape、预处理与论文描述一致性|
|Formula/Loss Auditor|`019f3fd5-14a8-7202-ae54-6ae7b7e1f545`|DANN、GRL、LMMD、Eq.16训练目标和损失语义|
|Protocol/Experiment Matrix Auditor|`019f3fd5-4f3a-7170-9e35-8fde2ed260c3`|ManySig协议、R比例、Fig.7/TableI矩阵和CVS边界|
|Fine-tuning/Sampling Auditor|`019f3fd5-826a-7b12-b0d7-4908a83bae9a`|Fig.8fine-tuning、不确定性采样、1/50budget和source replay|
|Code Reachability/Test Auditor|`019f3fd5-b658-7ae0-9846-56c2ca112a20`|CLI可达性、dry-run产物、测试覆盖和训练路径|
|Final Supervisor|`019f3fdd-7ea1-72b2-9dbd-bc3bf41e3892`|汇总互审结果，逐项对应论文内容并给出最终判定|

## 2.最终判定

当前工作不能称为“全方位复现完成”。当前最多可称为：paper-faithful闭集跨接收机UDA复现脚手架，已具备部分模型、损失、采样helper和dry-run矩阵。

正式复现仍被阻断，主要原因是：

1.正式训练入口缺失。
2.真实ManySig loader与论文预处理流水线缺失。
3.Eq.16LMMD第二阶段训练未接入训练循环。
4.Fig.8fine-tuning训练闭环缺失。
5.dry-run矩阵没有真实receiver split、seed/repeat、训练超参和正式指标。
6.非`--dry-run`路径在失败前仍可能写出dry-run payload，产物可信边界不清。

## 3.阻断与风险清单

|优先级|问题|当前判定|
|---|---|---|
|P0|`train.py`没有正式训练路径，非`--dry-run`只报错。|blocked|
|P0|论文预处理未落地：equalized time-domain preamble、power normalization、CFO preserved、first256IQ均缺少执行证据。|blocked|
|P1|Eq.16端到端目标`source CE+lambda*sum(LMMD_l)`只存在损失原语，未进入DANN后第二阶段训练。|blocked|
|P1|Fig.8fine-tuning只有采样/budget/replay helper，没有100iterations训练曲线和random sampling比较。|blocked|
|P1|Fig.7矩阵只有外层骨架，没有真实split、seed/repeat、超参、指标和TableI逐receiver目标字段。|partial|
|P1|非`--dry-run`失败前仍会构造并写出payload，应先gate再输出。|incorrect|
|P2|domain classifier采用1-logit+BCE，论文未严格指定该实现；traceability不能写成严格复现。|partial|
|P2|卷积kernel/channel schedule属于实现选择，不应宣称论文逐字指定。|partial|
|P2|LMMD缺target probability单纯形校验，missing-class batch语义与Eq.14固定`1/K`存在差异。|partial|
|P2|GRL lambda与`domain_weight`可能被误解为双重缩放，需要训练配置语义隔离。|partial|
|P2|测试缺GRL梯度符号、`dann_loss`、`multi_layer_lmmd_loss`、非dry-run不写output、真实WiSig split/loader和formal fine-tune。|partial|

## 4.论文方法逐项对应

|论文方法项|当前完成工作|状态|
|---|---|---|
|paper-faithful与CVS Stage2/LEO/open-set隔离|config为`cvs_extension=false`，protocol拒绝混入CVS扩展，claim blocks明确排除Stage2-C/LEO/open-set。|strict|
|ManySig 6TX/12RX/4days协议|config/dry-run记录6TX/12RX，未证明真实数据读取和split。|partial|
|预处理：equalization、power normalization、CFO preserved、256IQ|仅有config声明和模型输入转换，无执行证据。|blocked|
|输入shape`[batch,1,256,2]`|`_to_paper_conv_input`支持`[B,2,256]`、`[B,256,2]`和`[B,1,256,2]`。|strict|
|4个Conv-BN-ReLU-MaxPool、128feature|模型核心对齐；kernel/channel是实现选择。|partial|
|TX classifier/domain classifier|TX多类logits基本可用；domain为1-logit+BCE而非softmax，属实现选择。|partial|
|GRL+DANN损失方向|GRL接入、source TX CE、source/target domain BCE存在；缺训练循环和梯度符号测试。|partial|
|LMMD公式原语|source one-hot、target predicted probability、多层activation方向正确；缺概率校验和完整Eq.16训练落地。|partial|
|DANN后LMMD第二阶段训练|无可达训练阶段。|blocked|
|Fig.7实验矩阵|dry-run覆盖R比例和方法列；缺真实split、超参、seed和指标。|partial|
|TableI逐target receiver accuracy|仅写目标值，未生成正式结果。|deferred|
|Fig.8few-sample fine-tuning|采样/budget/source replay只是helper；正式训练和曲线缺失。|helper-only|
|正式测试与可达性证明|shape、LMMD backward、sampling、protocol基础测试存在；关键训练路径未测。|partial|

## 5.互审一致结论

五个审计在核心结论上相互印证：模型骨架与若干公式原语已搭出，但正式复现被正式训练入口、真实ManySig预处理/loader、LMMD第二阶段训练和fine-tuning训练闭环阻断。dry-run矩阵不能替代真实训练结果；sampling helper不能替代Fig.8复现；CVS Stage2/LEO/open-set边界隔离正确。

主要冲突来自现有traceability状态过满：

1.Formula/Loss Auditor指出Eq.16未落地，与RAU-04写`verified`冲突。
2.Fine-tuning/Sampling Auditor指出RAU-05只能算helper-only，与RAU-05写`verified`冲突。
3.Protocol/Experiment Matrix Auditor指出RAU-08只是dry-run骨架，与RAU-08写`verified`冲突。
4.Code Reachability/Test Auditor新增指出非dry-run失败前仍可写output，属于产物可信度问题。

## 6.traceability建议修正

`analysis/receiver_agnostic_twostage_uda_traceability_20260708.md`应在后续修订中调整为：

|条目|建议状态|
|---|---|
|RAU-01|partial|
|RAU-02|partial|
|RAU-03|partial|
|RAU-04|拆成LMMD原语partial和Eq.16训练blocked|
|RAU-05|helper-only|
|RAU-06|partial，并明确预处理执行证据blocked|
|RAU-07|partial，或注明仅核心结构strict|
|RAU-08|partial|
|RAU-09|deferred|
|RAU-10|拆成sampling helper-only和Fig.8formal blocked|
|RAU-11|strict|

还应新增CLI可达性问题：非`--dry-run`必须在任何payload写出前gate，或提供明确正式训练入口。

## 7.最小闭环清单

1.修复CLI gate：非`--dry-run`在任何payload写出前失败，或实现正式训练入口。
2.落地ManySig正式loader/preprocess：真实receiver/TX/day字段、equalized preamble、power normalization、CFO preserved、256IQ、split/seed/repeat记录。
3.实现Stage1DANN训练和Stage2LMMD Eq.16训练，记录TX loss、domain loss、LMMD loss、source/target accuracy。
4.补测试：GRL梯度符号、`dann_loss`、`multi_layer_lmmd_loss`、target_probs概率校验、非dry-run不写output、真实split smoke。
5.实现Fig.8fine-tune loop：uncertainty/random sampling、1/50budget进入DataLoader、source replay进入loader/loss、100iterations曲线。
6.产出正式复现报告：Fig.7矩阵结果、TableI逐receiver accuracy、Fig.8曲线，并继续声明该复现非CVS Stage2、非LEO部署、非open-set/new-class注册。

## 8.本轮执行边界

本轮审计为只读多agent对照审计。未访问N607，未启动训练，未修改复现代码，未修改现有dirty文件。新增本文件用于持久化多子agent审计结论。

## 9.后续修正状态更新

更新时间：2026-07-08后续审查。

以下早期审计问题已经由后续提交修正，不应继续作为当前阻断项引用：

|原问题|当前状态|
|---|---|
|非`--dry-run`失败前仍可能写出payload|已修复。`train.py`先gate正式训练，再允许dry-run payload写出；测试覆盖非dry-run不写output。|
|domain classifier使用1-logit+BCE|已修复。当前domain head输出2类logits，loss优先使用cross entropy以贴合论文softmax classifier表述。|
|LMMD缺target probability单纯形校验|已修复。`lmmd_loss`会校验target probabilities非负且逐行和为1。|
|缺GRL符号、DANN loss、非dry-run no-output测试|已补测试。|

以下仍是当前阻断或未完成项：

|论文复现项|当前状态|
|---|---|
|真实ManySig预处理/loader运行证据|仍blocked。本地未登记真实`ManySig.pkl`；同步/前导提取/信道均衡只能作为WiSig compact pkl上游前提。|
|Stage1 DANN与Stage2 LMMD正式训练|仍partial。已有单batch step和合成smoke测试，但没有epoch训练、checkpoint、真实receiver split或正式metrics。|
|Fig.8 fine-tuning正式曲线|仍helper-only。已有采样、1/50预算、source replay、合成batch和single-step测试；没有真实target labeled pool和100iteration accuracy曲线。|
|Fig.7/TableI正式结果|仍deferred。dry-run只保存论文矩阵和论文参考值，不是复现实验结果。|

后续提交说明必须继续限定为“paper-faithful复现骨架、协议、数据管线或训练原语修正”，不得写“全方位复现完成”或“达到论文结果”。
