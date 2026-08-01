# D111-LOO-GAT设计落地追溯

源设计：`analysis/d111_loo_gat_theory_20260802.md`

|ID|源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D111-B01|§2|定义仅允许INT8值＋FP16尺度的bundle schema与严格成员集|`code/cvsrffi/stage2_d111_loo_gat_bundle.py`|verified|NPZ/目录成员集正负测|不得沿用可替换v1/v2 sidecar|
|D111-B02|§2|绑定checkpoint、registry、method lock、来源aggregate、代码/配置与outer signature|同上|verified|Ed25519源签名、外层签名、身份漂移与篡改负测|release-controlled公钥；API不接收签名秘密|
|D111-B03|§2|禁止source row/ID/path、FP32中心、dense导出、query/truth与未签sidecar|同上|verified|公共API、成员白名单、protocol receipt测试|生成器只接收规范化aggregate|
|D111-B04|§3|由逐类\(B_c\)等权投影平均确定性导出共同Top3\(U\)|同上|verified|不同class projector的非恒等类置换测试|规范字节顺序累加保证类置换确定性|
|D111-B05|§3|封存量化扰动谱隙与\(U^TU\)证书，失败拒绝formal资产|同上|verified|退化谱隙负测、解码正交回执复算|不得替换rank或sidecar|
|D111-B06|§4|仅从Phase1 aggregate机械导出\(v_g,v_s,B,\epsilon\)及其provenance|同上|verified|FP32同副本哈希/计算、B/epsilon向上量化、类置换测试|不得用held性能选择|
|D111-B07|§2-§4|strict Phase2 loader只解码必要聚合，无dense/source导出或可写cache|同上|verified|深度readonly、签名/manifest/NPZ篡改测试|组件待封印；验签后才形成effective formal状态|
|D111-B08|§8|输出bundle数值字节、生成MAC、解码MAC和峰值数组上界|同上|verified|resource receipt构建/载入复算|`source_domain_count`由content root与outer seal绑定|
|D111-S01|§4|实现32步Weiszfeld、primal-dual gap与3/5共识资格|`code/cvsrffi/stage2_d111_loo_gat_score.py`|verified|dual可行、gap、共识、固定步数与严格回退测试|阻尼固定1/2|
|D111-S02|§5|实现等先验单位质量anchor/support凸混合|同上|verified|完整Student-t常数、`gamma=1`门禁、M0同带宽与单位质量公式测试|新类`rho=0`，无old bias|
|D111-S03|§6|正交坐标等变与物理局部界测试|同上|verified|类置换、坐标置换、稠密正交实数层测试|不声称INT8 bit级旋转不变|
|D111-G01|§7|真实588 tap的K1/K5/K10无truth G0|未授权|deferred|待实现、复审与另行预登记|bundle实现不得自动触发|
|D111-G02|§7|source-held性能实验|未授权|deferred|待G0三K功能变化|RPP与四臂合并已拒绝|

## 当前反向审计

- 当前实现范围为`D111-B01`至`D111-S03`，计数为`verified=11 / deferred=2 / rejected=1(RPP合并路线) / blocked=1(生产authority公钥与真实formal资产)`。
- `D111-G01`与`D111-G02`仍为有意延后；评分核完成不自动触发实验。
- 定向验证：`tests/test_stage2_d111_loo_gat_bundle.py`共10项通过；相邻知识包回归合计208项通过，唯一输出为既有PyTorch AMP弃用警告。
- 独立Terra Max复审：`P0=0 / P1=0 / P2=0 / CODE_ACCEPTANCE_GO`。
- 评分核与相邻知识包完整回归：217项通过；评分核独立Terra Max复审为`P0=0 / P1=0 / P2=0 / GO`。
- 最高风险项仍是生产keyring刻意为空。它使未登记authority的资产fail-closed；代码验收通过不等于真实formal资产已经生成，更不等于性能有效。
