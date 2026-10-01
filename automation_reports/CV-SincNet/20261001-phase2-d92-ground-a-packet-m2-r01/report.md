# D92适应前地面分类头：两个固定模型小包导出

- run_id：`20261001-phase2-d92-ground-a-packet-m2-r01`
- group_id：`d92-ground-a-source-packet`；类别：`analysis`；阶段：`Phase2-source-classifier-packet-export`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从两个既有合规source-only scratch final200 checkpoint提取原始6×160 float32 CosFace分类头，显式绑定原头类别行序、s=30、eps=1e-4和raw feat_joint ABI。原模型及训练不变；只建立后续配对A测量输入，本run不读取任何源域样本、逐样本源特征、目标support/query，不训练、不推断、不评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

输入绑定VERIFIED：原factory为model.build_model，原CosFace的s=30.0、eps=1e-4；实际类别行序为14-10、14-7、20-15、20-19、6-15、8-20。两个模型均source-only scratch final200，完整继承为空，按training_final_only选中。原始head数值部分每个应为3840B；完整包字节数在实际导出后测量。本run不测量A准确率、B−A或星载资源；增量传输量仍N/A。此前只读factory绑定缺arch_family的失败已保留，按原factory默认值与CLI实际合并路径修正，未构造encoder、加载checkpoint或启动训练。

LOCAL_VERIFIED/VERIFIED：原头导出31项、相关scorer/runner/publisher组合71项、final200修订后runner/publisher34项合成检查通过；首轮scorer失败保留并已修。独立P0/P1检查发现的原模型final200和生产档案minimalcontext兼容问题已闭合。原factory/class/s/eps及source-only scratch final200真实元数据已绑定，实际导出/目标推断/评分尚未执行；A与B−A仍N/A。
