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
