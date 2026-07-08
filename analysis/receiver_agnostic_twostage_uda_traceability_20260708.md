# Receiver-Agnostic Two-stage UDA复现逐项对应表

范围：Bao et al., "Receiver-Agnostic Radio Frequency Fingerprinting Based on Two-stage Unsupervised Domain Adaptation and Fine-tuning", IEEE GLOBECOM 2023。

边界：本表先做paper-faithful reproduction。CVS Stage2-A/B/C、satellite/LEO、`Y_old/Y_new/Y_unknown`和N607运行若出现，必须另标`cvs_extension=true`。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|RAU-01|Abstract, Sec.III-A|两阶段UDA：先DANN全局域对抗，再LMMD相关子域适配。|`paper_reproduction/receiver_agnostic_twostage_uda/model.py`; `losses.py`; `steps.py`; `train.py`|partial|模型/loss原语、Eq.16组合目标、Stage1/Stage2单batch step有单测；正式训练入口仍gate|正式epoch训练循环、checkpoint和真实ManySig长跑未落地，不能声明完整复现。|
|RAU-02|Eq.6-Eq.9, Fig.2|source有TX标签，target只有无标签；模型含feature extractor、TX classifier、domain classifier、GRL。|`model.py`; `losses.py`; tests|verified|模型shape、2类domain head、GRL符号测试|domain label按source=0,target=1。|
|RAU-03|Eq.10-Eq.13|DANN损失和GRL更新语义：TX CE + domain CE，feature梯度反转。|`losses.py`; tests|verified|DANN loss与GRL梯度符号测试|domain classifier按论文softmax表述改为2类logits。|
|RAU-04|Eq.14-Eq.16, Fig.3|LMMD按source one-hot label与target预测概率计算类别权重，支持多层activation；Eq.16组合source CE与LMMD。|`losses.py`; `steps.py`; tests|partial|LMMD backward、target_probs校验、Eq.16组合目标和Stage2单batch step测试|正式Stage2 epoch训练未落地；缺真实ManySig batch覆盖验证。|
|RAU-05|Sec.III-C, Fig.4|fine-tuning使用target不确定性采样得到少量标注样本，并混入少量source样本保持source性能。|`sampling.py`; `steps.py`; tests|helper-only|uncertainty/random排序、1/50预算、source replay、Fig.8 batch合成和单batch CE step测试|缺真实target labeled pool、source replay loader和性能曲线。|
|RAU-06|Sec.IV-A|WiSig ManySig，6 TX、12 RX、4 days；equalized time-domain preamble，power normalization，保留CFO，取前256 IQ并组织为`[batch,1,256,2]`。|`data.py`; `model.py`; config/report|partial|config/protocol校验4 days；合成ManySig loader契约测试覆盖`equalized=1`、`crop_mode=left`、256 IQ、source/target RX不重叠|本地缺真实`ManySig.pkl`，同步/前导提取/信道均衡只能作为上游compact pkl前提；正式数据长跑仍blocked。|
|RAU-07|Sec.IV-B|feature extractor为4个Conv-BN-ReLU-MaxPool block，输出128维；TX/domain classifier为Dense128+ReLU+softmax。|`model.py`; tests|partial|模型shape和activation数量测试通过|核心结构对齐；卷积核/通道数属implementation choice。|
|RAU-08|Sec.IV-C Fig.7|总体实验按source:target receiver比例`R`评估：source-only下界、target-labeled retrain上界、DANN、DANN+LMMD。|`configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json`; `train.py`; `protocol.py`|partial|CLI dry-run只生成论文矩阵骨架，并显式标`artifact_type=dry_run_only`、`formal_training_status=blocked`、`result_claim_status=no_formal_metrics`|Fig.7固定比例、TableI论文参考值、Fig.8迭代点和split/seed未公开边界已登记；正式训练结果未生成。|
|RAU-09|Sec.IV-C Table I|`R=6:6`逐target receiver accuracy表：0.89/0.94/0.87/0.91/0.92/0.92为论文参考对照。|future report|deferred||当前字段为`table_i_paper_reference_accuracy`且`reference_only=true`；需要正式训练/评估结果，当前不得声明达到。|
|RAU-10|Sec.IV-C Fig.8|`R<4:8`时比较fine-tuning不确定性/随机采样策略和迭代次数；标注样本数为unlabeled dataset的1/50。|`sampling.py`; `steps.py`; `protocol.py`; report|helper-only|sampling helper、random seed、Fig.8 batch合成、single-step fine-tune和100iteration报告点测试|真实100iteration曲线和accuracy结果待正式实现/验证。|
|RAU-11|项目协议|paper-faithful与CVS extension分离；不把target unlabeled UDA写成CVS source-only DG或Stage2成功。|README; config; `protocol.py`|verified|protocol测试拒绝`cvs_extension=true`混入|硬门槛。|

## 当前阻断与风险

- 原文未给完整优化器、batch size、epoch、GRL系数、LMMD kernel细节和fine-tuning冻结策略；这些字段必须标为`paper-unspecified`或`implementation choice`。
- 本轮尚未运行N607；所有结果声明只能到本地代码/测试、合成smoke和复现实验设计层。真实ManySig预处理证据、Stage1/Stage2正式epoch训练、Fig.7/TableI/Fig.8结果仍未完成。
- 本轮验证命令：
  - `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_receiver_agnostic_twostage_uda.py`
    - result: `16 passed`
  - `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile paper_reproduction/receiver_agnostic_twostage_uda/model.py paper_reproduction/receiver_agnostic_twostage_uda/losses.py paper_reproduction/receiver_agnostic_twostage_uda/sampling.py paper_reproduction/receiver_agnostic_twostage_uda/protocol.py paper_reproduction/receiver_agnostic_twostage_uda/train.py paper_reproduction/receiver_agnostic_twostage_uda/data.py paper_reproduction/receiver_agnostic_twostage_uda/steps.py tests/test_receiver_agnostic_twostage_uda.py`
  - `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m paper_reproduction.receiver_agnostic_twostage_uda.train --config paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json --dry-run --output local_artifacts/receiver_agnostic_twostage_uda_dry_run_20260708.json`
