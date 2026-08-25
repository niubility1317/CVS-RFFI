# PA-M2.1设计—实现—实验追踪

来源：用户批准的《PA-M2.1独立因子复审与后续优化实施计划（修订版）》及对应设计规格。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|M21-01|§1、§3|候选命名和theta迁移声明边界|设计、runner、报告|verified|manifest/report检查|不得称正确challenge系统辨识|
|M21-02|§2.1|`V_audit_retro`仅称权重独立|设计、报告|verified|报告复核|非研究历史完全未见|
|M21-03|§2.2|同容量C1′/C4′控制|runner、factor artifact|verified|参数量/step/init测试|同初始化模板|
|M21-04|§2.3|覆盖4个holdout fold|核心模块、runner|verified|4-fold/raw-mask测试+正式factor artifact|正式run四fold全部闭合|
|M21-05|§2.4|support只来自独立bank|核心模块|verified|role来源测试|审计目标不得充当support|
|M21-06|§2.4、§7|关系选择不读取q|核心模块|verified|无q接口及候选测试|metadata后确定性seed|
|M21-07|§4.1|保持0.07/0.63/0.15/0.15|runner|verified|协议负测|只细分V_select|
|M21-08|§4.1|V_select按65/35或覆盖不足时70/30|核心模块、manifest|verified|split复现测试+正式split manifest|实际8001/4176/423|
|M21-09|§4.2|TX×RX×day×eq×block分组|核心模块|verified|block不跨role测试|B从10/20/25元数据选择|
|M21-10|§4.2|guard block不进入role|核心模块|verified|相邻role隔离测试|降低邻近重复|
|M21-11|§4.3|exact/near duplicate聚合审计|核心模块、artifact|verified|聚合字段测试|不返回样本级摘要|
|M21-12|§4.3|sig_i不得解释为跨RX同步ID|核心模块、报告|verified|语义字段与F7测试|F7保持UNAVAILABLE|
|M21-13|§5|sidecar V3完整architecture_config|核心模块、artifact|verified|round-trip测试|含无参数语义|
|M21-14|§5.2|V3严格加载和contract核对|核心模块|verified|token/stride/contract负测|strict state load|
|M21-15|§5.2|V2仅显式迁移challenge encoder|核心模块|verified|legacy模式负测|不能直接输出V2|
|M21-16|§6|冻结Core90和旧challenge encoder|runner|verified|requires_grad/state检查|response/operator重新初始化|
|M21-17|§6.1|C1′/C4′只在L_s训练|runner|verified|role receipt|U_s不增加监督|
|M21-18|§6.1/6.2|只用V_select_fit选epoch|runner|verified|选择访问测试|audit不参与选模|
|M21-19|§7|F0–F9完整矩阵|核心模块、runner|verified|逐row/head训练测试+正式factor artifact|F7按预登记保持UNAVAILABLE|
|M21-20|§7.1|F3严格same TX/cross RX/same day|核心模块|verified|无fallback测试|无候选则invalid|
|M21-21|§7.2|F5严格different TX/same RX/day|核心模块|verified|关系测试|禁止循环shuffle|
|M21-22|§7.3|F6固定连续PA统计|核心模块、runner|verified|非q选择与距离测试|runner已接入8项固定物理统计|
|M21-23|§8.1|common-anchor与all-valid并报|核心模块、artifact|verified|mask/macro测试|核心集合F2∩F3∩F5|
|M21-24|§8.2|F3四项通过条件|判定模块|verified|边界值测试|5%、5%、10%、80%|
|M21-25|§8.3|C4′对C1′至少3%且CI>0|判定模块|verified|三分支verdict测试|A_PASS/PARTIAL/FAIL|
|M21-26|§9|3个head seed、3个mapping seed、2个sat seed|runner、artifact|verified|seed闭合测试|主sidecar seed 1个|
|M21-27|§10.1|q条件、shuffle、DeepSets、ordered probe|核心模块、runner|verified|probe schema测试|固定条件子集|
|M21-28|§10.2|M0精确检索指标|核心模块、artifact|verified|手算rank测试|候选池同TX/RX/day/fold|
|M21-29|§10.3|码本只诊断不做gate|runner、report|verified|decision输入测试|不优化hard均衡|
|M21-30|§11|6折leave-one-TX公共模型|核心模块、runner|verified|held-out TX负测|旧HR不作结论|
|M21-31|§11.2|residual probe与距离审计|runner、artifact|verified|schema/有限值测试|含between-TX和same-TX cross-RX|
|M21-32|§12|阶段B仅A_PASS运行|runner、launcher|verified|状态机测试|其他状态NOT_RUN_A_GATE|
|M21-33|§12.1|有界残差融合|核心模块|verified|g=0/全拒绝/clip测试|eta仅0.05/0.10/0.20|
|M21-34|§12.2|gate特征禁止truth/TX/RX/day|核心模块|verified|allowlist负测|只部署可用字段|
|M21-35|§12.3/12.4|V_cal group-CV拟合并冻结|runner|verified|role不交叉测试|audit标签不参与|
|M21-36|§12.5|B的7项安全门槛|判定模块|verified|边界值测试|含coverage和worst RX|
|M21-37|§13|两阶段分层结论|decision manifest、报告|verified|verdict测试|精确route命名|
|M21-38|§14|TDD逐项先红后绿|测试记录|verified|93项相关pytest通过|每个新增行为先观察失败再实现|
|M21-39|§15|14个聚合artifact|runner、报告|verified|artifact闭合测试|gate未运行也有状态文件|
|M21-40|§16|新不可覆盖run和最小发布治理|launcher、报告|verified|N607归档SHA、远端编译、真实checkpoint smoke、正式artifact闭合|旧A/B全程只读；C自然闭合ANALYZED|
|M21-41|复审治理|逐文件SHA、环境锁不作gate|报告|rejected|`REJECTED_EXTRA_GATE`记录|仅一次release归档SHA|
|M21-42|后续边界|Soft-DTW/OT/码本均衡/多机制/Core90解冻|报告|deferred|范围复核|不进入M2.1|

## 当前计数

- `pending`：0项
- `implemented`：0项
- `verified`：40项
- `deferred`：1项
- `rejected`：1项
- `blocked`：0项

最终结论：F2∩F3∩F5共同anchor覆盖为100%，F3相对F0和F5均有正增量，但C4′相对F2的跨receiver退化为12.79%—14.71%，超过预登记10%上限。阶段A为`A_FAIL`，阶段B按计划`NOT_RUN_A_GATE`，当前PA theta迁移路线停止；这不是系统技术失败。
