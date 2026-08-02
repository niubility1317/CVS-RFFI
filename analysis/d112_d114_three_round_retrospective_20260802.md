# D112–D114三轮轻型域适应研发回顾

状态：`RETROSPECTIVE_COMPLETE / D114_G1_NEGATIVE / D114_CLOSED`

日期：2026-08-02

## 1.回顾范围与证据边界

本回顾在D116设计或任何新性能矩阵前完成。已重新读取当前目标与`项目.md`，并在`ssr-gpu`环境将项目conversation index刷新到1261条记录，检索`D112 D113 D114 SEAM BCAT HBPD G0 G1 功能 负结果`。正式判断仍以本地Git报告、完整artifact和同row结果为准；conversation index只用于防止重复路线。

D114的N607 G0随后完成，source-held四臂G1也已完整评分。D114在base与head背景下均为明显负收益，因此本回顾追加终态裁决；仍不把G0变化数量解释成收益。

## 2.三轮同行证据

|轮次|理论作用对象|最接近真实性能／功能证据|裁决|可保留知识|
|---|---|---|---|---|
|D112-SEAM|用target old support估计共享球面运动并移动Phase1 ground anchor；单位质量head负责输出|source-held G1完整63行／189单元。K1中ground head相对M0：old BA`+1.3228pp`、seen-new`+1.3228pp`、H`+1.9736pp`、old floor`+4.5855pp`；但joint与head在63/63行prediction完全相同|关闭SEAM motion，不调rank/收缩/rho；保留`M_HEAD_GROUND`|静态ground单位质量head有正收益；共享球面运动虽数值非零，却没有独立决策效应，不能把head收益记为DA|
|D113-BCAT|从六旧类support估计全类共享加性接收机偏移，统一逆变换support和query|真实588行无truth G0：K1/K5/K10均改变588条feature、score、margin；argmax变化为`0/1/0`|`REJECT_REVISION_NO_FUNCTION`，不发布N607/G1|“数值变化”不足以证明有效适应；共同加性平移在K1/K10没有跨过决策边界，继续放大只是在调强度|
|D114-HBPD|以sealed旧类条件方差和new池化先验替换qKNN经验带宽，直接建模support-query预测不确定性|本地真实588行无truth G0：K1/K5/K10的argmax变化为`16/73/29`，query fit/update=0，truth=false；N607性能尚无|允许冻结G0复现；只有N607一致后才做四臂G1|直接改变每类密度峰值与尾部比公共坐标平移更能进入决策路径；但功能非零仍不等于old/new/H/floor正收益|

## 3.失败模式归纳

1.分类head与DA必须严格拆开。D112证明ground head可以正收益，也证明SEAM motion独立效应为0；任何后续joint提升都必须同时给出`DA_AT_BASE`和`DA_AT_HEAD`，不能用`JOINT−M0`代替DA贡献。
2.共同变换容易产生“大量连续值变化、几乎零决策变化”。D113已否定当前共享加性平移，不再研究其强度、收缩、投影或高阶展开。
3.仅把source nuisance统计边缘化并不自动构成target DA。D115的receiver-marginal PLDA posterior predictive经监督后被判定为分类head替换：没有target receiver posterior，也无法构造独立DA因素，因此永久关闭为本轮DA路线。
4.全矩阵或方向性metric不是优先路线。D93/D94的ground→target transport覆盖低且为负，D110的低秩Fisher/PSD metric损害old/new/H/floor；下一方法不能用“换rank或调强度”重新包装。
5.K1是可识别性核心。无target类内自由度的方法必须从sealed Phase1 aggregate与跨类support结构获得合法状态；support自margin、类内方差或query自适应都不能在K1冒充可靠域证据。

## 4.协议与目标复核

- 继续使用`p2_min_v1`、固定`leo_*_weak`单观察、support/query物理ID不交叉；query零fit、零update、零selection。
- Phase2只读sealed int8 Phase1 aggregate、当前row support、registry与固定配置；禁止clean/source运行时访问、query truth/role、类quota和跨query重排。
- 每条query独立面对全部注册类；append new类不得改变既有类状态。方法必须同时保护old adaptation与new registration，而不是只优化一侧。
- 下一性能证据必须同row报告注册前后old、seen-new、H、per-class old floor与forgetting；不以单receiver、单K、边际最大值或局部正行晋级。
- D114若进入G1，只运行冻结`M0/M_DA/M_HEAD/M_JOINT`必要矩阵；不启动125，不补seed，不根据G0的16/73/29调参。

## 5.下一轮裁决

D114已经触发关闭条件：`DA_AT_BASE`与`DA_AT_HEAD`均无独立正收益，且K1总正确数、H与old floor系统下降。HBPD不进入Target25或125，也不做浓度、先验或带宽扫描。

D114若关闭，下一DA候选必须同时满足：显式估计target域状态而不是只换分类head；K1可由跨类support识别；不是公共加性平移；不是D110式共享PSD/Fisher metric；不是D114式带宽/浓度重参数化；与`M_HEAD_GROUND`形成可辨识四臂。尚未满足这些条件的想法只保留为理论草案，不实现、不发布实验。
