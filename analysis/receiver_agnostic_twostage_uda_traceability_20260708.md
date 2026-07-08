# Receiver-Agnostic Two-stage UDA复现逐项对应表

范围：Bao et al., "Receiver-Agnostic Radio Frequency Fingerprinting Based on Two-stage Unsupervised Domain Adaptation and Fine-tuning", IEEE GLOBECOM 2023。

边界：本表先做paper-faithful reproduction。CVS Stage2-A/B/C、satellite/LEO、`Y_old/Y_new/Y_unknown`和N607运行若出现，必须另标`cvs_extension=true`。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|RAU-01|Abstract, Sec.III-A|两阶段UDA：先DANN全局域对抗，再LMMD相关子域适配。|`paper_reproduction/receiver_agnostic_twostage_uda/model.py`; `losses.py`; `train.py`|implemented|`pytest tests/test_receiver_agnostic_twostage_uda.py` passed|干跑入口仍是gate，不执行正式长跑。|
|RAU-02|Eq.6-Eq.9, Fig.2|source有TX标签，target只有无标签；模型含feature extractor、TX classifier、domain classifier、GRL。|`model.py`; `losses.py`; tests|verified|模型shape测试通过|domain label按source=0,target=1。|
|RAU-03|Eq.10-Eq.13|DANN损失和GRL更新语义：TX CE + domain BCE，feature梯度反转。|`losses.py`; tests|implemented|`py_compile` passed|后续可补GRL梯度符号单测。|
|RAU-04|Eq.14-Eq.16, Fig.3|LMMD按source one-hot label与target预测概率计算类别权重，支持多层activation。|`losses.py`; tests|verified|LMMD backward测试通过|目标标签不被要求为真实标签。|
|RAU-05|Sec.III-C, Fig.4|fine-tuning使用target不确定性采样得到少量标注样本，并混入少量source样本保持source性能。|`sampling.py`; `train.py`; tests|verified|uncertainty ranking测试通过|冻结策略仍为paper-unspecified。|
|RAU-06|Sec.IV-A|WiSig ManySig，6 TX、12 RX、4 days；equalized time-domain preamble，power normalization，保留CFO，取前256 IQ并组织为`[batch,1,256,2]`。|`model.py`; config/report|partial|模型入口`[batch,2,256]`转换测试通过|正式数据split/loader长跑未执行。|
|RAU-07|Sec.IV-B|feature extractor为4个Conv-BN-ReLU-MaxPool block，输出128维；TX/domain classifier为Dense128+ReLU+softmax。|`model.py`; tests|verified|模型shape和activation数量测试通过|卷积核/通道数属implementation choice。|
|RAU-08|Sec.IV-C Fig.7|总体实验按source:target receiver比例`R`评估：source-only下界、target-labeled retrain上界、DANN、DANN+LMMD。|`configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json`; `train.py`|verified|CLI dry-run生成`local_artifacts/receiver_agnostic_twostage_uda_dry_run_20260708.json`|长跑矩阵已生成，未启动N607。|
|RAU-09|Sec.IV-C Table I|`R=6:6`逐target receiver accuracy表：0.89/0.94/0.87/0.91/0.92/0.92为目标复现对照。|future report|deferred||需要正式训练/评估结果，当前不得声明达到。|
|RAU-10|Sec.IV-C Fig.8|`R<4:8`时比较fine-tuning不确定性采样策略和迭代次数；标注样本数为unlabeled dataset的1/50。|`sampling.py`; `train.py`; report|partial|sampling测试通过|fine-tune训练循环和100 iteration曲线待长跑实现/验证。|
|RAU-11|项目协议|paper-faithful与CVS extension分离；不把target unlabeled UDA写成CVS source-only DG或Stage2成功。|README; config; `protocol.py`|verified|protocol测试拒绝`cvs_extension=true`混入|硬门槛。|

## 当前阻断与风险

- 原文未给完整优化器、batch size、epoch、GRL系数、LMMD kernel细节和fine-tuning冻结策略；这些字段必须标为`paper-unspecified`或`implementation choice`。
- 本轮尚未运行N607；所有结果声明只能到本地代码/测试和复现实验设计层。
- 本轮验证命令：
  - `conda run -n ssr-gpu python -m pytest -q tests/test_receiver_agnostic_twostage_uda.py`
  - `conda run -n ssr-gpu python -m py_compile paper_reproduction/receiver_agnostic_twostage_uda/model.py paper_reproduction/receiver_agnostic_twostage_uda/losses.py paper_reproduction/receiver_agnostic_twostage_uda/sampling.py paper_reproduction/receiver_agnostic_twostage_uda/protocol.py paper_reproduction/receiver_agnostic_twostage_uda/train.py`
  - `conda run -n ssr-gpu python -m paper_reproduction.receiver_agnostic_twostage_uda.train --config paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json --dry-run --output local_artifacts/receiver_agnostic_twostage_uda_dry_run_20260708.json`
