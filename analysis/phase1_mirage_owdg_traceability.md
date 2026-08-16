# Phase1 MIRAGE-OWDG可追溯记录

范围：Task 1——同步已批准的Phase1科学协议。

本记录在修改协议正文或实现前建立。批准的歧义解决优先于任务简报和设计规格中仍保留的`0.07/0.63/0.30`及文档逐字断言写法：执行接口采用`0.07/0.63/0.15/0.15`，并用行为测试证明权限边界。

| ID | 规格来源 | 需求 | 实现文件 | 测试 | 状态 | 证据与说明 |
|---|---|---|---|---|---|---|
| T1-POL-01 | 用户批准歧义解决；设计规格§3.2 | Phase1 source划分固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。 | `code/cvsrffi/phase1_mirage/protocol.py`；两份协议正文 | `tests/phase1_mirage/test_protocol_policy.py` | verified | RED为缺少模块；GREEN聚焦测试8项通过。 |
| T1-POL-02 | 用户批准歧义解决；设计规格§2-3 | `L_s/U_s/V_s`可以共享source-known TX身份；所有角色的物理样本ID必须互斥。 | `code/cvsrffi/phase1_mirage/protocol.py`；两份协议正文 | `tests/phase1_mirage/test_protocol_policy.py` | verified | GREEN验证允许共享TX并拒绝重复`physical_sample_id`。 |
| T1-POL-03 | 用户批准歧义解决；设计规格§3.3 | `proxy_train`只能源自`L_s`，且允许参与拒识相关梯度。 | `code/cvsrffi/phase1_mirage/protocol.py`；两份协议正文 | `tests/phase1_mirage/test_protocol_policy.py` | verified | GREEN验证唯一`L_s`来源和唯一拒识梯度权限。 |
| T1-POL-04 | 用户批准歧义解决；设计规格§3.3、§6.2 | `P_cal`只能源自`V_cal`并只可校准；`P_select`只能源自`V_select`并只可选模。 | `code/cvsrffi/phase1_mirage/protocol.py`；两份协议正文 | `tests/phase1_mirage/test_protocol_policy.py` | verified | GREEN验证来源与唯一权限，validation proxy无训练状态权限。 |
| T1-POL-05 | 用户批准歧义解决；设计规格§3.1、§3.4 | target unknown TX与source训练/validation TX身份互斥；所有target角色不得训练、校准、选模或触发选择性重跑。 | `code/cvsrffi/phase1_mirage/protocol.py`；两份协议正文 | `tests/phase1_mirage/test_protocol_policy.py` | verified | GREEN分别覆盖source train和validation重叠，并枚举两种target角色的全部权限。 |
| T1-DOC-01 | 用户任务边界；`E:/type10-7/AGENTS.md`版本管理规则 | 根目录`项目.md`与Git承载面的`docs/PROJECT_PROTOCOL.md`表达同一Phase1规则；根目录不提交。 | `E:/type10-7/项目.md`；`docs/PROJECT_PROTOCOL.md`；Task 1报告 | 语义marker审计；`git diff --check` | verified | 两份正文均检出批准语义，旧proxy禁训/TX固定互斥文本已移除；根目录为非Git镜像。 |
| T1-F1-C1 | Task 1审查finding C1 | 根正文和报告必须使用稳定Git镜像相对路径，记录隔离分支/commit，并声明主检出仅在最终集成后可见；临时worktree绝不成为协议权威。 | `E:/type10-7/项目.md`；Task 1报告 | 两份协议语义只读对照 | verified | 根正文固定`github_publish/CVS-RFFI-repo/docs/PROJECT_PROTOCOL.md`、隔离分支与实现提交`50cf8fde`；未修改主检出副本。 |
| T1-F1-I1 | Task 1审查finding I1 | target-known与target-unknown必须明确绑定相同单物理样本单LEO weak观测、预处理、前向与决策规则。 | `E:/type10-7/项目.md`；`docs/PROJECT_PROTOCOL.md` | 两份协议语义只读对照 | verified | 两份正文逐句检出相同观测、预处理、前向与决策边界。 |
| T1-F1-I2 | Task 1审查finding I2 | `CANDIDATE_RERANK`成为显式权限；两种target角色均必须fail closed。 | `code/cvsrffi/phase1_mirage/protocol.py` | `tests/phase1_mirage/test_protocol_policy.py` | verified | RED为缺失枚举；GREEN验证两种target角色均拒绝该权限。 |
| T1-F1-M1 | Task 1审查finding M1 | target unknown TX与source train/validation TX完全不重叠时必须通过身份检查。 | `code/cvsrffi/phase1_mirage/protocol.py` | `tests/phase1_mirage/test_protocol_policy.py` | verified | GREEN验证完全不重叠集合返回成功。 |
| T2-DATA-01 | Task 2简报；设计规格§3.1-3.2；批准policy接口 | 构建器只在source inventory阶段读取TX truth；按`TX×receiver×day`和`physical_sample_id`确定性划分`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，复用`Phase1DataPolicy.partition_counts`且物理ID严格互斥。 | `code/cvsrffi/phase1_mirage/data.py` | `tests/phase1_mirage/test_data.py` | verified | RED为8项缺失`cvsrffi.phase1_mirage.data`失败；GREEN为数据测试8项通过，含同seed稳定、异seed变化和同TX跨分组。 |
| T2-DATA-02 | Task 2简报；设计规格§3.1-3.2；`docs/PROJECT_PROTOCOL.md` Phase1地面数据 | target receiver在任何划分前fail closed；重复物理ID也必须拒绝。 | `code/cvsrffi/phase1_mirage/data.py` | `tests/phase1_mirage/test_data.py` | verified | GREEN使用带重复物理ID的target row验证target receiver先于拆分/重复检查拒绝；source重复ID也独立拒绝。 |
| T2-DATA-03 | Task 2简报；设计规格§3.2 | `UnlabeledView`结构上不得含有`tx_label`；Labeled/Validation视图只暴露各自获批字段；manifest提供ID哈希、分组计数和source TX/receiver登记。 | `code/cvsrffi/phase1_mirage/data.py`；`code/cvsrffi/phase1_mirage/__init__.py` | `tests/phase1_mirage/test_data.py` | verified | GREEN验证`UnlabeledView`字段、属性和序列化均无`tx_label`；Labeled/Validation字段、角色校验和四分区ID哈希回执均通过。 |
| T2-FR1-P0 | Task 2 fix round 1；控制器审查finding P0 | source split必须要求调用方显式提供权威receiver约束；省略约束时fail closed。 | `code/cvsrffi/phase1_mirage/data.py` | `tests/phase1_mirage/test_data.py` | verified | RED验证无`forbidden_receivers`的调用未抛异常；GREEN验证必需keyword-only参数，数据测试12项与限定回归22项通过。 |
| T2-FR1-P1 | Task 2 fix round 1；控制器审查finding P1 | 三个物化入口必须接收manifest，并只接受其固定分区的物理ID；拒绝`U_s→Labeled`与`V_cal→val_select`。 | `code/cvsrffi/phase1_mirage/data.py` | `tests/phase1_mirage/test_data.py` | verified | RED验证当前入口未要求manifest且接受跨角色ID；GREEN验证manifest必需、`U_s→Labeled`和`V_cal→val_select`均fail closed，限定回归22项通过。 |
| T3-PROXY-01 | Task 3简报；设计规格§3.3、§5.4、§10；fix round1/I1 | 只从获批的labeled split生成确定性、周期平衡且逐episode标签置换等价的source proxy episode；按首次出现row位置形成匿名类组顺序；proxy类在registered logits/prototypes前移除；收据只保留置换不变统计；全部异常输入与未获批角色fail closed。 | `code/cvsrffi/phase1_mirage/proxy.py` | `tests/phase1_mirage/test_proxy.py` | verified | 初始RED为缺少`proxy`模块的17项失败，GREEN为17项通过。I1新增不均衡类数同一episode置换测试，RED为proxy rows不一致，GREEN为18项通过；限定Task1-3回归40项通过。实现复用`Phase1DataPolicy`的`proxy_train/P_cal/P_select`唯一origin语义，不复制权限表。 |
