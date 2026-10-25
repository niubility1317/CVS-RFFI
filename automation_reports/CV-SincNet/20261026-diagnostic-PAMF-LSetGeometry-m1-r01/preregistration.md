# 可见L的单中心与多样本参照几何诊断

PLANNED。本run是冻结源L诊断，不训练新F，不访问U/V/目标，不把非参数参照分数当作替代分类头或部署结果。原三种子raw clean≥+1pp、raw六景>+1.5pp、同G-LU六景超过两个对照的目标保持。

## 比较与数据

只用原F0可见L6300、同源三冻结own E20+10的E30、同原G-LU。分别在每个教师自己的160维坐标保存L样本，按默认头类别/RX组织，不能跨seed平均坐标。query仅seed2026092701的冻结E30。原source合同与VALIDATED_ONCE句柄保持，U56700/V27000不打开；读取旧visibleL refs.pt只用于数值和计数等价检查。

Mean沿用每类/RX单位均值。Set05/Set10保留全部单位L向量，每类/RX内分数为tau*logmeanexp(cos(z,p)/tau)，tau分别.05/.1；其他RX等权平均，排除query自身RX，最后统一除以外温.1。logmeanexp使重复参照数量不改变分数；不以某个TX单独定规则。Set10是主要参照形式，Set05仅作已固定敏感性比较，全部结果保留。

原L全部6300条逐一检查raw、G(raw)、G(low suburban)、G(mid urban)、G(low urban)五视图。H物理生成沿既有source_channel，namespace PAMF/set_reference/L/fixed，channel/hardware seed20261016421/20261016423、residual/post_sync/noEQ/fs25e6/fc2462e6/SNR30/reflect。每包同一冻结默认头预测，参照与默认头的可见真标签CE只在独立detached feature处求梯度，不向F/G参数写grad。报告全部按TX类别索引/RX/day和overall的参照正确率、默认头正确率、错配、特征梯度cos/有效计数，以及L类内平均合成向量长度。它们是几何证据，不是参数更新、未见RX迁移或目标收益保证。

## 来源、资源与停止

加载前复用原E20/ownE30来源与相同sourcecontract检查，禁止目标训练/选模父模型。实际E30/G、L2、默认头重算和特征兼容smoke是第一个执行步骤，通过立即继续。旧Mean中心与新bank同源坐标/计数一次核对，阈值maxabs≤1e-5；不重验数据或增加哈希/签名层。所有F/G参数、persistent/nonpersistent registeredbuffers必须不变，参数.grad全部None。

N607现有普通账户Python/Torch2.1，FP32/TF32off，不安装依赖。GPU0仅一条本run，准入时计入全部CUDA/pending<2且free≥10000MiB，CPU2threads，保护其他owner。执行命令与CWD见experiment.json；不可覆盖run/release，代码身份为runtime.code_commit。一次archive SHA传输比较、remote compile，启动后独立PID/CWD/完整命令/GPU/loggrowth检查。短任务如在观察前完成，明确NOT_CAPTURED，用最终artifact验证，不为观察窗口重跑。

任何协议、来源、smoke、非有限、状态更新或技术失败终止本run，保留全部产物，不自动重提、不热改或停止其他任务。新训练为0，历史三个E30共享90唯一F轮；query复用seed01，不重复计父成本。新F完整前向50400包、G25200包，另有L2smoke；500次feature autograd，不做encoder/optimizer更新。私有bank、IQ、逐样本分数与runtime不提交Git。

几何改善不能让旧三变体晋级。后续若研究新SSL，仍先按已固定+.3pp/- .1pp源留出条件验证，再按冻结矩阵进行真实三种子raw/同G目标测试。本run只报告全部几何结果，不自动启动训练、目标预测或更换默认方法。

发布前本候选唯一一次独立 P0/P1 审查 PASS，无 P0/P1；4 条聚焦检查和相关模块 compile PASS。真实 checkpoint L2 smoke 由 launcher 首步执行，通过后立即继续。
