# Stage2-C sparse key-layer delta protocol mirror

`E:\type10-7`根目录不是Git仓库。2026-07-15根协议`E:\type10-7\项目.md`增加了一个由用户明确授权的窄例外：Stage2-C可使用`support-only sparse key-layer delta`对ADV3B02少量预注册原层进行快速校准。本文是Git承载面交接，科学场景和数据权限仍以根协议为准。

硬约束：

- 精确层名白名单，禁止模糊扩展；
- 原checkpoint可训参数≤50,000；
- target-support校准≤5epoch且≤50个optimizer step；
- 星上SGD无momentum，不部署optimizer状态；
- FP16差分补丁+分类head总状态≤128KB；
- 只读注册target support；query不参与拟合、选模或早停；
- 禁止query真old/new角色、类别配额、全批类别数量先验和Hungarian分配；
- 必须单列更新的原参数量、差分状态和部署新增MAC，不得冒充为完全冻结adapter。

首个预注册白名单是`id_backbone.t_proj`、`id_backbone.f_proj`和`id_backbone.pa_proj.0`；`id_backbone.fuse.0`单层已超出本路线的50k总参数预算，不在白名单内。
