# 地面训练双路线

地面训练有两个版本：集中式训练版和元学习版。两者的训练组织不同，但协议边界完全相同：只能使用源接收机域`R_s`，身份标注比例满足`rho_label<=0.1`，无TX标签样本只能在source-domain weak-label/semi-supervised DG框架内使用。

## 集中式训练版

集中式训练版把`R_s`内的弱标注样本和无TX标签样本汇聚到一个地面训练流程中。它的目标不是few-shot episode accuracy，而是学习跨接收机稳定的`z_id`身份空间，并通过`z_dom`、GRL/leakage probe、domain supervision、prototype agreement和source-only satellite stress控制identity-style conflict。

入口：

```powershell
python code\train.py --train_mode centralized --dataset wisig --wisig_pkl <path-to-wisig-pkl> --wisig_train_ratio 0.1
```

推荐在正式训练中记录：

| 字段 | 要求 |
|---|---|
| `source_receivers` | 只来自`R_s` |
| `target_receivers` | 地面训练阶段不得使用 |
| `rho_label` | 不超过`0.1` |
| `domain_label` | receiver/day/rx_day/channel等，可用于域监督和采样平衡 |
| `ssl_audit` | 伪标签precision、coverage by class/receiver、quota和uncertainty |
| `leakage_probe` | `z_id -> receiver`泄漏探针 |
| `satellite_stress` | deployment-oriented validation-control，不是部署成功 |

集中式训练版适合做强基线、ablation和主干模型选择。任何使用`R_t`统计或验证信号的训练都不能再标为source-only。

## 元学习版

元学习版在`R_s`内部构造episodic source split，用receiver、day、rx_day或其组合模拟未见接收机外推。episode中的support/query都必须来自源域；`R_t`不得作为inner/outer query、验证域、early stopping域或threshold域。

入口：

```powershell
python code\train.py --dataset wisig --wisig_pkl <path-to-wisig-pkl> --use_meta_ssl_cvs --use_meta_rxday_episodes --wisig_train_ratio 0.1
```

协议检查入口：

```powershell
python code\train.py --dataset wisig --wisig_pkl <path-to-wisig-pkl> --use_meta_ssl_cvs --meta_ssl_protocol_check_only --use_meta_rxday_episodes --wisig_train_ratio 0.1
```

相关代码：

- `code/cvsrffi/meta_episodes.py`
- `code/dataset_wisig.py`
- `code/tests/test_meta_ssl_cli_defaults.py`
- `code/tests/test_meta_ssl_split.py`
- `code/tests/test_meta_ssl_train_loop.py`

论文复现中的`paper_reproduction/protonet_cda/`提供episodic ProtoNet基线，但它是论文复现/对照层，不自动等同于CVS地面元学习主线。若把它作为CVS主线对照，应在报告中标注其baseline身份和协议差异。

## 共同禁止项

- 不得用目标接收机域`R_t`做训练、模型选择、BN统计、阈值拟合或prototype初始化。
- 不得把地面阶段写成部署few-shot。
- 不得把clean view成功写成satellite/LEO deployment success。
- 不得只报告单个指标最大值；主结论必须绑定candidate/run、split、K、receiver/TX划分和完整同row指标。
