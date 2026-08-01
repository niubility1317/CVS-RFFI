# D111-LOO-GAT设计落地追溯

源设计：`analysis/d111_loo_gat_theory_20260802.md`

|ID|源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D111-B01|§2|定义仅允许INT8值＋FP16尺度的bundle schema与严格成员集|待定新bundle模块|pending|成员集正负测|不得沿用可替换v1/v2 sidecar|
|D111-B02|§2|绑定checkpoint、registry、method lock、来源aggregate、代码/配置与outer signature|待定新bundle模块|pending|SHA/signature漂移负测|formal资产必须fail-closed|
|D111-B03|§2|禁止source row/ID/path、FP32中心、dense导出、query/truth与未签sidecar|待定新bundle模块|pending|禁止成员/接口负测|必须在生成器和loader两侧拒绝|
|D111-B04|§3|由逐类\(B_c\)等权投影平均确定性导出共同Top3\(U\)|待定新bundle模块|pending|类置换/符号/重放一致性|basis不得被直接冒充为U|
|D111-B05|§3|封存量化扰动谱隙与\(U^TU\)证书，失败拒绝formal资产|待定新bundle模块|pending|谱隙/正交性正负测|不得替换rank或sidecar|
|D111-B06|§4|仅从Phase1 aggregate机械导出\(v_g,v_s,B,\epsilon\)及其provenance|待定新bundle模块|pending|确定性/量化误差/无sample输入测试|不得用held性能选择|
|D111-B07|§2-§4|strict Phase2 loader只解码必要聚合，无dense/source导出或可写cache|待定新bundle模块|pending|API面、readonly、tamper测试|只授权bundle实现|
|D111-B08|§8|输出bundle数值字节、生成MAC、解码MAC和峰值数组上界|待定新bundle模块|pending|资源receipt单测|不宣称未实测RSS|
|D111-S01|§4|实现32步Weiszfeld、primal-dual gap与3/5共识资格|未授权|deferred|待bundle formal后单独授权|当前不编写分数core|
|D111-S02|§5|实现等先验单位质量anchor/support凸混合|未授权|deferred|待bundle formal后单独授权|严禁old logit bias|
|D111-S03|§6|正交坐标等变与物理局部界测试|未授权|deferred|待score core实现|不声称bit级旋转不变|
|D111-G01|§7|真实588 tap的K1/K5/K10无truth G0|未授权|deferred|待实现、复审与另行预登记|bundle实现不得自动触发|
|D111-G02|§7|source-held性能实验|未授权|deferred|待G0三K功能变化|RPP与四臂合并已拒绝|

## 当前反向审计

- 当前实现范围仅`D111-B01`至`D111-B08`。
- `D111-S01`至`D111-G02`均为有意延后，不能在bundle实现中偷渡。
- 最高风险是历史v2只处于`PENDING_OUTER_JOINT_SEAL`；实现只能构建新joint-sealed D111资产，不能将历史件重标为formal。
