# 特征增强版多解耦设计追溯

来源：用户附件 `8f00af9d-39ef-43d2-9af5-ab77f7803c39/已粘贴的文本.txt`。用户要求严格实现并发布三种子，继续纯 CE、无半监督和星地训练增强。报告中旧 U/EMA/LEO 配方由明确授权覆盖：Fishr 仅真实 clean，条件权重为 1，不生成伪 LEO 观测。保持 L/U/V 数据角色，U 不读。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| F01 | III.1 VII | scratch E200、L128、222步、44400更新、纯CE/cosine无U教师星地 | design/source | verified | native integration | 不改健康旧任务 |
| F02 | III.2 | 独立B48：6TX×2RX×4uniquePID、同day、30blocks公平轮换 | sampling | verified | coverage checks | 每8原生step |
| F03 | III.2 | 每格无放回遍历再洗牌、不足换block且记录、不同PID不冒充独立采集 | sampling | verified | insufficient/cycle tests | collection元数据缺失明示 |
| F04 | III.3-5 | 类比例一致、仅L真标签、R独立三袋不共池 | runtime/sampling | verified | isolation negative tests | randomB48对照 |
| F05 | IV.1 | 状态条件L/T、跨recipient迁参数重新计算、exact29真实重算 | runtime | verified | inherited action checks | R不虚构exact29 |
| F06 | IV.1 | 小候选真实IQ核验、合法困难IQ不因代理误差被丢弃 | runtime | verified | proposal mismatch check | 降低代理可靠度 |
| F07 | IV.1 | 不重复同端点虚拟CE、条件状态/顺序rollout | runtime | verified | route checks | 保留L/T/R固定权重 |
| F08 | IV.2 | time_down唯一弱MixStyle、statdetach、Beta.1/p.15/eta.2 | style | verified | equation and gradient tests | 不改PA/reference/z |
| F09 | IV.2 X | donor同TX不同RX同day不同PID、缺donor不增强 | style | verified | predicate tests | 标签不进分类头margin |
| F10 | IV.2 VI X | Style仅native L CE、无重复CE/KL，按role隔离并清理、旧schedule不增强 | runtime/style | verified | native hook tests | D/Fishr/V关闭 |
| F11 | IV.2 X | 分支梯度与V关闭时间贡献margin探针 | runtime/validation | verified | real adapter probe | 非严格因果贡献 |
| F12 | V.2-4 | raw和direction正确cosine梯度[B,6,160]scale30、完整图不unitize | fishr | verified | autograd/formula checks | 固定960坐标 |
| F13 | V.1,5 | 样本轴n分母方差、两RX同日同条件、pair差平方mean/4 | fishr | verified | two-domain nonzero tests | 非logit代理 |
| F14 | V.3,4 | raw缩放a^-4、direction正行缩放不变 | fishr_checks | verified | numerical regression | 正常范数区间 |
| F15 | V.6 | 每桶beta.9、独立计数去偏、old detach/current graph、缺桶不更新 | fishr | verified | state/counter tests | checkpoint m/count/step |
| F16 | V.7 VI | Fishr真实IQ回传E/G/C、无style/virtual/pseudo、V不改state | runtime | verified | gradient routing checks | clean-only |
| F17 | VII | E1-20关系CE和统计，E21-40style/F升权，保留Adam和cosine | runtime | verified | schedule boundary tests | E131仍pureCE |
| F18 | VII.3 | 成熟桶源L梯度定标3/5/10%，随后固定，记录实际加权梯度/夹角 | runtime/fishr | verified | calibration checks | 非loss自适应增权 |
| F19 | VIII IX.1 | native/random B48/关系CE/raw5/direction5全三seed；5550关系calls | design/checks | verified | config/budget checks | clean曝光266400包次 |
| F20 | IX.2-3 | 同多解耦增强×Fishr核心四组；source选ratio后noR/shared/direct真实增强 | design/dispatch | verified | selection/target isolation | 48训练36测试 |
| F21 | IX.3 | shared容量匹配、相同动作/候选/视图/预算 | capacity/runtime | verified | parameter/route tests | 禁止假匹配 |
| F22 | X | macro/RX/day/TX/最差RX/margin低尾/质量分层/桶年龄/采样/style/作用R使用 | runtime/validation | verified | artifact assertions | 稀疏指标独立次数 |
| F23 | X.6 | sampler/loader/donor/Beta/action私有随机流不改变native序列 | sampling/style/runtime | verified | RNG isolation | 无LEO RNG |
| F24 | 发布 | 逐行冻结→clean+6practical→独立truth-last评分；源候选只测选中 | dispatch/evaluate | verified | negative gate tests | 不回流target调参 |
| F25 | 发布 | 完整登记、真实smoke、独立P0/P1、Git及N607证据 | checks/register/publish | implemented | 本地/独立审查通过；待N607发布读回 | 唯一launch owner |
| F26 | III.4 VII | 困难重采样作为后续独立项 | report | deferred | 报告明确首轮公平覆盖 | 不在Fishr流引入偏置 |
| F27 | IV.2 | DSU作为互斥替代候选 | report | deferred | 报告明确首轮弱MixStyle | 不叠加统计增强 |
| F28 | IV.1 | 剩余交互稳定后再加第四网络 | report | deferred | 条件未触发 | 本轮保持L/T/R |

预登记固定 13 种配置×3seed；其中 SF/SAF 的 3%、5%、10% 在源 V 完整三seed上选择共同ratio，只选中SF/SAF接触target；其余四候选保留source-only。随后该ratio的noR/shared/direct三种机制对照各三seed从零训练。共48次完整训练，36行测试。source-only选择不读取已有或本轮target成绩。

本地核验：24项verified，F25已实现等待远端读回，3项按设计deferred，0项rejected/blocked。CPU/CUDA证据见本run evidence；不把实现检查表述为泛化收益。

私有流：动作model_seed+49071、关系采样model_seed+88739、Style donor/Beta model_seed+77917。Fishr定标使用E20全部成熟关系批次的梯度范数比中位数；E20末固定，之后不随惩罚下降增权。
