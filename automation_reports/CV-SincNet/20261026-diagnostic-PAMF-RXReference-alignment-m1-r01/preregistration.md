# 冻结源模型的身份特征梯度一致性

PLANNED。上一单种子sourceHold三参照变体均未通过预定+.3pp留出G增益，当前不扩展该矩阵。新run只区分源剂量不足与辅助目标不匹配的可能原因，不更新任何模型，不读取V或目标IQ/标签，不解封U真值。

## 来源与固定矩阵

固定同一个student seed2026092701的五个实际reader：原ordinary-PL E30、匹配PLControl F50、RawReference F50、RestoredReference F50、PairedReference F50。E30/PLControl依既有reader核对完整own E20/10/20来源；三参照F50核对当前sourceHold配置、own父链、实际checkpoint final_only/4440新增步、原默认头及native U/EMA关闭。源contract与物理角色沿原F0 VALIDATED_ONCE，不重复数据验证或添加哈希链。

参照复用已由可见L6300构造的own E30固定TX/RX特征，U概率与接受集合来自同一份原raw三E30教师缓存。缓存IQ为同原G-LU产生的三个源恢复场景；本run不新生成G输出或训练G。source context只有既有五个源句柄键，不含target/evaluation输入。

loader seed2026092701，12批L128/U256。L仅使用raw与可见真实标签onehot；U使用raw与同物理缓存G视图，k=(step−1)%3，依次low suburban/mid urban/low urban。U软目标为冻结raw概率q，硬目标仅为argmax(q)，不能称为U真值。

## 测量

每次只将冻结F的实际160维特征复制为可求导叶节点z，真实默认头参数仍冻结且eval。分别求参照关系CE、默认头soft-q CE、默认头hard-identity CE对z的梯度。参照温度.1，排除同RX参照，公式与此前训练完全相同。特征梯度不是完整encoder梯度，也不是AdamW方向。

记录逐包梯度余弦、范数、参照/硬身份梯度范数比与头目标概率。分层为all_seen、accepted、accepted的留出RX1/参与RX，以及默认头与身份目标一致/不一致。G三场景分别保留，并汇总U_G_ALL；汇总不增加前向。U仅报告与冻结伪标签的一致性，L准确率也是训练源样本诊断，不是泛化结果。

余弦有效条件为双方范数>1e−12；0样本或0有效梯度保留n=0与None，不虚构均值。共五模型×六统计视图×四组×三一致性层＝360个单元，所有单元保留。源特征梯度关联不能单独证明辅助目标无用、恢复身份丢失或某个新方法必定提高目标性能；本run没有自动扩展/选新超参数规则。

测量先求逐包损失的特征梯度，再按接受/留出分组汇总；当前并未执行原训练的留出mask更新。特别是RX1组描述假设施加该目标时的梯度关系，原sourceHold训练中它实际不进入新增损失。所有样本损失只通过各自特征求导，无跨样本目标或全局重分配。

预算：L1536/U3072曝光，五F×(1536+2×3072)＝38400次包前向、540次特征autograd；真实L2 smoke另列，G前向0。最后所有模型参数和全部registered buffers（包括非持久bank）不变，参数.grad仍None，optimizer、新头拟合、参照更新、V/目标读取均为0。

## 发布

launcher首步对真实L2检查五个defaulthead的features→classify_features与实际forward一致，并验证特征梯度有限；通过立即继续。三项聚焦数学/冻结边界检查PASS，相关模块compile PASS；发布前一次独立P0/P1审查结果追加。

命令：`PYTHON -u -m experiments.pamf_channel_pl.receiver_alignment_source dispatch --config PROJECT/releases/20261026-diagnostic-PAMF-RXReference-alignment-release-r01/runtime.json`。N607既有普通账户Python/Torch2.1，FP32/TF32off，CWD为该不可覆盖release，输出 `PROJECT/runs/20261026-diagnostic-PAMF-RXReference-alignment-m1-r01`。GPU0一条本run worker，准入时全卡CUDA/pending<2且空闲≥10000MiB，CPU2threads，保护所有其他owner。

Git commit为代码/config身份，一次archive SHA传输比较、远端compile、启动后独立PID/CWD/argv/GPU/loggrowth核实。首个来源/角色/真实head-smoke/非有限/模型状态变化或技术错误停止本run，保留部分产物，不自动重提或操作其他健康进程。预期smoke.json、progress.json、complete.json、pipeline_state.json及同run公开全分层源分析；它不能替代原三种子正式目标验收。

发布前一次独立P0/P1审查PASS，无P0/P1；核对五reader实际加载、特征叶节点求导、NumPy转换边界、源权限和矩阵计数。三项聚焦检查与模块compile PASS。真实L2 checkpoint/feature-gradient smoke由launcher首步执行。
