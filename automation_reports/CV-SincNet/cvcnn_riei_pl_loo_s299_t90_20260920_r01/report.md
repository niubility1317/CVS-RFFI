# STAR同配置CVCNN/RIEI普通伪标签对照

状态：RUNNING / VERIFIED，已发布并独立核实训练进行中。用户指定seed299、阈值0.9，两方法共2项，各200轮。

数据复用STAR E纯Loo基准的固定数组与划分代码：源RX0/1/2/3/5，目标RX4；TX0–9。输入为配对clean/Loo的(2,2048)时频表示。source全池9%有标签训练、90%无标签训练、1%验证；无标签TX在训练加载器前清除。split=299+rx。RX4仅eval模式预测，不参与BN更新、伪标签或梯度。该STAR对比任务按用户指定数据契约执行，不替换为CVS的CORE90数据默认值。

复用STAR E的拼接增强：32个有标签对、256个无标签对，展平为64/512视图；监督身份损失为0.5×CE_clean+0.5×CE_Loo。50%固定/50%动态Loo，增强seed91001，动态sigma第1–50轮从1到3，rho及scatter数值和极端衰落训练修复均逐字复用。没有额外CFO/IQ。固定source验证和RX4测试完全不做在线增强。

普通伪标签为每个clean/SG视图各自softmax argmax，detach后置信度≥0.9保留，对选中视图取平均交叉熵，乘0.65，自E30启用。无EMA教师、温度校准、历史一致性、soft标签、0.13 SG权重或梯度投影；也不使用STAR特殊PL余弦爬坡。最后推理EMA仅用于最终评估，不参与伪标签。

CVCNN直接使用STAR Ni_model_2part的复数编码器、后128维身份特征及确定性cosine均值分类头，冻结未使用sigma，关闭SNN/RD/orth。RIEI使用CVS baselines/riei_fd/model.py、architecture.py、losses.py，ResNet1D18及EC/RC，保留交替更新、MI=1.2、IE=1.2，feature norm=0.0001（已向用户说明采用CVS优化版的暂定解释）。SG附加目标只计算身份CE；接收机CE及解耦目标使用clean源数据，包括合法源无标签RX元数据，不用无标签TX真值。

优化与预算对齐STAR：Adam初始基准LR0.001，前10轮0.2→1倍，E30–200余弦到0.1倍；梯度clip10；采样seed92001+epoch，zip两loader。RIEI仍有额外FED解耦step，因此计算量并非相同，实际耗时独立记录。每10轮固定诊断，最终第200轮EMA为主、student辅助，EMA自E151逐step decay0.99。没有best选择、早停或按目标结果重跑。

模型来源：全部scratch，无pretrain/resume/teacher继承；EMA仅从本次student创建。核对结论SCRATCH_NO_INHERITED_STATE。

验证：verify.py在ssr-gpu CUDA验证硬标签阈值/无选中梯度、E29关闭和E30启用、拼接CE权重、clean增强不变、两个真实网络有限前后向、参数更新及checkpoint读回。详细见verification.json。独立P0/P1审查无阻断项；本地代码commit668711e；N607发布SHA与编译通过，远端CUDA验证通过，启动读回通过。

输出：manifest.json定义独立remote runs目录；batches.jsonl记录每批实际CE/PL/RIEI及增强；metrics_epoch.jsonl记录训练耗时、显存和诊断。最终预测先写盘，再由score读取预测连接truth，生成accuracy、Macro-F1、混淆矩阵。COMPLETE.json须全部终评产物写完后生成。单seed结果不宣称跨seed稳定性或显著性；已有STAR目标数据用于历史探索，本对比不宣称盲测确认。

部署：dispatch.py为唯一owner，独占launch.lock；全GPU所有用户进程纳入每卡最多2任务，优先空卡。失败停排队不杀健康任务，不按性能停机。保留历史所有实验和数据。此STAR Git检出没有remote，记录本地commit，不虚构push。

## 启动核验

- cvcnn：GPU0，PID142136；实际配置和5个源RX划分逐项匹配STAR。
- riei：GPU1，PID142613；实际配置和5个源RX划分逐项匹配STAR。

两项均已有完成epoch及持续增长的batch日志，CE/SG权重已实际启用；普通PL在E30启用，当前早期核验尚未启用。远端配置核验见config_audit.json；PID/CWD/命令/GPU/log见remote_inspect.json。未完成E200，不报告最终性能。GitHub初次推送因TLS失败，保留本地提交；最终交付状态见git_delivery.json。

用户修正：RIEI行已停止并由riei_pl_loo_ce11_s299_t90_20260920_r02从零替代，身份CE由0.5+0.5改为1+1。旧RIEI保留为partial，不作为修正版结果；CVCNN继续原配置。远端调度器原生FAILED状态表示用户纠正停机，不是数值/性能失败。
