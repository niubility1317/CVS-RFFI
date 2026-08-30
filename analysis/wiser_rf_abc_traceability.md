# WISER-RF ABC设计落地追踪

设计来源：`E:\codex\home\attachments\f9cea5b7-87cc-4da9-a19e-f84aff39767d\pasted-text.txt`

当前实现目标：在同一发布中实现A（WB-FT）、B（量化Phase1分布摘要+VSW）和C（模型反演诊断），其中A/B属于正式`p2_min_v1`主线，C保持隔离诊断，不参与正式晋级。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| W-01 | 协议修订 | 允许随checkpoint冻结的不可逆int8聚合Phase1分布摘要进入Phase2 | `项目.md`、`docs/PROJECT_PROTOCOL.md` | verified | 根协议与Git镜像均已加入5.3.2；摘要加载器实行穷尽成员白名单 | 不开放source replay、样本级embedding或伪源IQ |
| W-02 | 路线B/第六节 | 加载并验证量化域×类聚合中心或低秩源中心、域残差基、系数、半径 | `code/cvsrffi/wiser_source_summary.py`、`configs/wiser_rf_adv3b02_source_binding.json` | verified | 真实ADV3B02资产加载为`[6,14,160]`；运行前核对checkpoint/摘要身份、特征schema、维度和有序类表 | 正式ADV3B02使用5,363B域×类int8组件 |
| W-03 | 第十节第一步 | 冻结源分类权重，无target head，执行WB-FT双重监督+L2-SP | `code/cvsrffi/stage2_wiser_rf.py` | verified | `tests/test_stage2_wiser_rf.py`5项通过 | source-head使用无margin推理logits |
| W-04 | 第六/九节 | 从量化摘要确定性构造虚拟源特征并计算class-wise sliced-Wasserstein | `code/cvsrffi/wiser_source_summary.py` | verified | VSW确定性、有限值、target-only梯度及全stage固定投影seed测试通过 | 不持久化反量化float源库 |
| W-05 | 路线C | 实现模型反演伪源IQ诊断臂 | `code/cvsrffi/wiser_model_inversion.py` | verified | `tests/test_wiser_model_inversion.py`2项通过 | `DIAGNOSTIC_MODEL_INVERSION_NON_FORMAL`，不参与正式晋级 |
| W-06 | 第十一节 | Stage0映射检查；Stage1至Stage3渐进解冻；Sinc首轮冻结 | `code/cvsrffi/stage2_wiser_rf.py` | verified | 精确参数名、冻结范围和grad reach测试通过 | domain分支和辅助头始终冻结 |
| W-07 | 第十三节 | 生成P1冻结源头、P2冻结源原型、P3 old-only D92及表示几何prediction | `code/cvsrffi/stage2_wiser_runner.py` | verified | query冻结负测通过；真实ADV3B02`ABC`三阶段无query smoke通过并独立回读 | P4仅补充诊断 |
| W-08 | 第十三/十五节 | 独立truth-last scorer计算三probe、floor和类内/类间散度 | `code/cvsrffi/stage2_wiser_scoring.py` | verified | scorer强制精确query token注册表、行数、有限特征及6旧类覆盖 | truth不得回流训练/选择 |
| W-09 | 第十四节 | 同一pilot发布B0、A、B、C、ABC五臂 | `code/scripts/run_stage2_wiser_pilot.py` | verified | CLI、不可覆盖、“全部支持集状态先冻结、再首次打开query”及真实ABC smoke通过 | C/ABC-C标记非正式诊断 |
| W-10 | 第十五节 | 三LEO场景最小pilot；达到门槛后才扩大 | `automation_reports/CV-SincNet/<run-id>/report.md` | pending | N607 artifacts+独立评分 | 不因低性能终止已运行任务 |
| W-11 | 发布 | 提交、push、远端OID核对、唯一release SHA、远端编译与PID绑定 | Git/N607/report | pending | 独立readback | 不覆盖旧run root |

## 已核实资产规模

- 正式身份特征：`z_id=160`。
- pilot旧类数：6。
- 现有量化Phase1组件：`int8_domain_class_center_lowrank_residual_radius_v2.npz`。
- 实测文件大小：6,323B；数组payload共7,328B。
- 内容：6×160 int8中心、6×3×160 int8域残差基、13×6×3 int8域系数、14×6 int8半径及FP16尺度。
- ADV3B02匹配资产：`int8_domain_class_prototypes.npz`，实测5,363B；6类×14个有效source域中心×160维。
