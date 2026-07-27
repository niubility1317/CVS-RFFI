# CVS-RFFI IEEE Transactions英文初稿

日期：2026-07-27

状态：`EVIDENCE_LOCKED_INITIAL_DRAFT / NOT_SUBMISSION_READY`

本目录给出可继续修改和正式编译的IEEE Transactions双栏英文初稿。正文结构、摘要长度、关键词、编号引用、表格和参考文献样式按IEEE期刊稿组织；当前作者信息保持匿名。

## 文件

- `manuscript.tex`：规范主稿源文件。
- `references.bib`：36条已核验真实文献。
- `manuscript.pdf`：由Tectonic 0.17.0编译的审阅PDF。
- `claim_evidence_matrix.md`：逐项主张、证据和禁用表述。
- `citation_inventory.md`：36份引用PDF的来源、页数、解析结果和SHA-256清单。

## 当前可写结论

1. Phase1的`ADV3B02_CORE90_SOFT_E200`属于source-only闭集DG证据：`overall=89.18%`、`strict UDU=84.89%`、`receiver_floor=75.55%`。
2. RTB-IDR（D92）完成125/125 matched诊断；在`K=10,new20`上相对matched control的注册后旧类准确率提高2.622个百分点、最低旧类准确率提高4.600个百分点、`H_old_new`提高0.964个百分点，但新类准确率下降0.653个百分点。
3. RTB-IDR状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它不能被写成已达到部署目标或已解决Stage2。
4. WiSig/ManySig是地面代理数据，`leo_*_weak`是物理启发压力代理；不得写成真实在轨实验。

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
- 补数据manifest、物理记录数、采样率和LEO弱信道参数。
- 若继续使用“spaceborne”定位，至少补硬件在环；若主张真实卫星有效性，则必须补真实卫星数据。
- 补目标处理器上的端到端时延、峰值内存、能耗和数值一致性。

36份引用PDF放在桌面交付目录，不纳入Git仓库，以避免把受版权约束或体积较大的论文文件提交到代码版本库。
