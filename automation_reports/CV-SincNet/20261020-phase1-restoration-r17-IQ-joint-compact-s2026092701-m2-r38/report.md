# R38：无 TX 标签全链压缩与性能联合优化

状态：**VERIFIED / RUNNING**。N607 两版已通过真实权重无 query smoke 并完成首轮训练；七场景正式测试尚未完成。完整参数1709713/794529为结构计数，未证明正式性能或加速。方法 commit `e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381` 已推送并独立核实；35 项聚焦检查、CLI help、一次 scoped P0/P1 通过；实际最终模型与完整测试尚待执行。实际发布、真实权重 smoke、首轮训练和所属进程读回已完成。


固定两档 dense 通道收窄，不采用增加卷积分解层的低秩推理图：`dense_three_quarter` 完整1709713参数（较4665073减少63.3508%），`dense_half` 完整794529参数（减少82.9686%）。含coarse/action/prior，分别context1165243/prior544470及context548795/prior245734。卷积与线性MAC、FFT/CG之外开销、完整恢复延迟和显存实测单列，不能由参数减少推定加速。

原source-selected R17 unroll_unshared/E20/4940/cd977448为唯一教师与初始化来源，原L6300/U56700/V及F50/defaulthead契约保持；不继承R04/R05/R06/R07/R08压缩学生或R32权重。两档宽度与目标在读取本轮R37完整正式分数前固定，只依据原R17源权重结构、先前source侧共享归一化保真诊断和低秩源延迟测量作设计，不以目标分数选择/重跑。

teacher保留自身原coarse/action上下文。学生每层按原GroupNorm四组等量取通道，原Conv/Linear/GN对应参数切片，decoder输入严格映射为原upstream索引＋位于原全宽偏移处的skip索引；不能把拼接后前几个通道简单裁掉。原三步prior长卷积weight取均值并共享，保留各步bias/GN/1x1，source输入物理9通道与最终2通道保持。全初始化0 IQ/TX/F/V/target/优化更新，非函数等价或无损剪枝声明。

学生coarse、action和prior全部可训练。估计上下文直接调用coarse和estimator，绕过历史ActionRestorer.estimate中的coarse no_grad；教师全部冻结且停梯度。仍三步prior×两次CG，推理只接收当前IQ、输出一份IQ；原PAMF从同一输出派生FFT/PA/时域/统计。无TX/RX/场景路由、clean/教师/F推理输入、跨帧或状态更新。

优化目标为 final paired-clean IQ NMSE＋λ(epoch)×三个stage到原R17 IQ教师的NMSE均值，分母为detach clean能量floor1e-4。λ固定E1=10线性降至E20=1。配对clean确实进入优化，区别于只拟合教师的精简；教师只作源IQ保真锚点，最终性能必须由正式测试证明。0 TX标签、0训练好F权重或输出、0 V/target优化。名称为TX无标签模拟配对自监督，含源监督分类器的完整系统不称全部无监督。

两行相同全R07已验证缓存63000=L6300/U56700，20个cached received epoch、15%clean/85%残余信道、原顺序/数据seed2026102001/model2701。E20/4940，batch256/micro64按真实最后batch分母累积、每globalbatch一次AdamW和clip1；LR2e-4/warm247/cos2e-5、WD1e-4、FP32/noTF32。每行L126000/U1134000，teacher和student各1260000输入（teacher每输入原coarse/action/三prior）。原teacher各组件20轮、新学生各组件20轮，各累计40；合计120组件epoch，按训练阶段串行80pass，与固定原F50/E50成本分别列明。无optimizer状态进入部署profile。

每行训练完整后自动source七视图、同源16样本batch1成本测量和独立CPU源诊断；source不筛行、不早停、不改参数。每行随后自动原168000物理样本×clean/三suburban/三urban正式预测；该行全部IQ/features/logits/variance/ID验证固定后，独立CPU truth-last。固定原F50 seed2026092701及原默认头，0新拟合分类头。每场景较原R17最多少1680正确（1pp），按整数判断；这是退化容限，目标为更小网络下实际正收益。lowurban70%及其他五环境距rawclean≤5pp、clean损伤另报，不能以场景均值掩盖超限。

N607普通账户、不可覆盖新root/固定Git release，GPU0/1各一ownworker、总计最多两条。先等待R37根终态且实际ownedGPU句柄退出，不读取其分数；共享receiver_residual_capacity4.lock、source至少18GB空闲及全owner每卡少于两进程。保留其他owner，CPU2线程/本run score.lock最多一独立scorer。launcher首步实际原R17/两学生/原F50/2Ls CPU无query smoke，检查所有组件梯度、教师/F不变、全checkpoint严格复载输出逐位一致，PASS立即继续。F只在该无query读出和冻结评测使用，不进入loss或source训练模块。

一次聚焦协议测试和独立P0/P1、一个Git代码/config身份、最小登记、一次普通账户预检/一个archive两端SHA比较/远端compile、启动后PID/CWD/argv/GPU/日志产物读回。输入/权限/冲突/非有限数/技术错误时保留产物及健康行，不自动重试，不把低分当技术错误。每行完成全部测试/分析/登记后才称该实验完成；单F或source教师拟合不证明新增细节属于输入TX。当前已实际发布并核实两版训练运行，尚未证明本路线的正式性能。


精确命令、环境、路径和逐行矩阵见[登记](experiment.json)。


## 训练已启动

VERIFIED：实际原R17/两学生/冻结F50 no-query smoke PASS；教师/F不变，各组件梯度有效。GPU0/1各一worker的PID/CWD/argv/CUDA及首epoch/日志增长已核实；两档全链联合训练正在运行，尚无正式性能结论。每版固定E20后自动七场景正式测试，源分数不筛行。
