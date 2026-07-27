# CVS-RFFI IEEE Transactions英文初稿

日期：2026-07-28（信道建模与前置章节修订）

状态：`EVIDENCE_LOCKED_INITIAL_DRAFT / NOT_SUBMISSION_READY`

本目录给出可继续修改和正式编译的IEEE Transactions双栏英文初稿。正文结构、摘要长度、关键词、编号引用、表格和参考文献样式按IEEE期刊稿组织；当前作者信息保持匿名。

## 文件

- `manuscript.tex`：规范主稿源文件。
- `references.bib`：38条已核验真实文献。
- `manuscript.pdf`：由Tectonic 0.17.0编译的审阅PDF。
- `claim_evidence_matrix.md`：逐项主张、证据和禁用表述。
- `citation_inventory.md`：38份引用PDF的来源、页数、解析结果和SHA-256清单。

## 2026-07-28修订

- 修正Phase1总损失公式为加权和，补齐所有加号。
- 重写引言与任务建模，明确跨接收机迁移、少样本新类注册和全类独立判决的复合任务。
- 用中性的clear-sky、low-elevation和rainy-link名称呈现三种LEO星地信道场景，论文正文不暴露协议内部场景键。
- 新增实现一致的LEO残余基带信道方程、公共参数表和场景参数表；明确绝对自由空间路径损耗、显式大气衰落和额外IQ不平衡未写入当前IQ波形。
- 新增3GPP TR 38.811与Abdi等人的阴影Rician信道文献，并将PDF加入桌面引用文件夹。

## 当前可写结论

1. Phase1的`ADV3B02_CORE90_SOFT_E200`属于source-only闭集DG证据：`overall=89.18%`、`strict UDU=84.89%`、`receiver_floor=75.55%`。
2. RTB-IDR（D92）完成125/125 matched诊断；在`K=10,new20`上相对matched control的注册后旧类准确率提高2.622个百分点、最低旧类准确率提高4.600个百分点、`H_old_new`提高0.964个百分点，但新类准确率下降0.653个百分点。
3. RTB-IDR状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它不能被写成已达到部署目标或已解决Stage2。
4. WiSig/ManySig是地面代理数据，LEO信道属于仿真残余基带模型；不得写成真实在轨实验或完整星地链路预算。

## 编译

标准IEEE LaTeX环境可执行：

```text
pdflatex manuscript
bibtex manuscript
pdflatex manuscript
pdflatex manuscript
```

本次审阅PDF使用便携Tectonic编译：

```text
tectonic -X compile manuscript.tex --keep-logs --keep-intermediates
```

正式投稿前应使用目标期刊推荐模板重新编译，并处理正文中的全部`[AUTHOR ACTION: ...]`。

## 投稿前硬缺口

- 补冻结的`p2_min_v1`正式确认矩阵和same-row完整主表。
- 补同capsule、同seed、同新类规模的ProtoNet、单qKNN、adapter-qKNN和最新主候选。
- 补Phase1独立复验、置信区间和模块级消融。
- 补数据manifest和各split物理记录数；采样率与LEO信道参数已进入正文。
- 若继续使用“spaceborne”定位，至少补硬件在环；若主张真实卫星有效性，则必须补真实卫星数据。
- 补目标处理器上的端到端时延、峰值内存、能耗和数值一致性。

38份引用PDF放在桌面交付目录，不纳入Git仓库，以避免把受版权约束或体积较大的论文文件提交到代码版本库。
