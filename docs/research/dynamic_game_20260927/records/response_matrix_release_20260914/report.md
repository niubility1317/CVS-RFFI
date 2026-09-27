# 实验矩阵发布记录

用户在第二轮验收后明确要求“发布实验矩阵”，本轮据此进入发布与执行。此前“未启动”的报告保留为准备阶段历史状态。

发布范围：127份既有实体配置全部进入固定代码release；首批执行36行，为新路线SIM/EG/CF-EG/TR-EG/XT-DANN×3seed的18行，以及前轮R0四格＋ADV0两格×3seed的18行。其余90份新路线配置等待源域选择；旧R1参考及新旧依赖模板保留，不擅自填数或启动。三模型seed392005/6/7，split/data/augmentation保持392005。训练预算固定E200×222接受步，优化器、LR、损失及机制不因发布改变。

全部从零训练，不读取任何历史模型/EMA/原型。仅复用既有`runs/phase1_adv3b02_xuc15_s392005_20260913_r1/source_contract.json`物理角色契约：L_s6300/U_s56700/V27000；训练入口逐项比较实际role_ids。数据集为`Dataset_WigSig/ManySig.pkl`。source V只读，选模遵循既有规则。首批训练后停在`SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW`，不自动扩大矩阵或访问目标数据；目标预测及独立评分在源域选择冻结后进入后续阶段。

发布前只读预检：N607普通账户szu2070436088，主机dell-DSS8440，8张RTX3090无计算PID，无活动训练/dispatcher，/home剩余约7.1TB；Torch2.1.0+cu121、CUDA可用。唯一launch owner是本轮主Agent；每卡最多两任务，含尚未建立CUDA上下文的训练进程预留。容量不足则队列等待。单一dispatcher顺序提交；不与其他dispatcher争抢。任一技术失败暂停后续行，健康在跑行自然完成；不因低性能停止，不自动重试，不覆盖已有run-root。

本地算法验收沿用113项PASS，不重复测试未变算法。新增控制面核对36唯一ID、scratch、显式execute、完成需TRAINING_COMPLETE＋E200＋44400步及最终checkpoint。无query检查脚本执行四条实际响应路径完整L128/U256，并逐条保存/严格重建诊断checkpoint；这些权重仅用于检查，永不作为正式初始化。远端使用现有CVS-RFFI环境，不安装pytest。传输校验、编译、检查成功后继续提交唯一dispatcher；状态需要独立PID/CWD/argv/GPU/log读回，launch返回值不等同于运行成功。

独立P0/P1审查指出GPU多dispatcher竞态和旧smoke不覆盖新路线；前者以本次唯一活动launch owner及发布前现场读回闭合，后者已替换为`check_response_execution.py`。远端状态、检查结果及准确run/release路径由后续读回记录补充；本文件不预先声明RUNNING或性能结论。

## 发布后独立读回：VERIFIED_RUNNING

- run：`phase1_native_response_matrix_20260914_r1`；dispatcher PID476257。
- 运行代码：`c9c0bb314fa09fc6e77e07282d7adfbf96b31298`，已push并独立核对origin OID。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_response_c9c0bb314f`。
- run-root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_native_response_matrix_20260914_r1`；同名logs目录记录训练日志。
- 四条CUDA真实路径及checkpoint重建PASS；检查中峰值分配显存CF/TR/XT各约2.84GiB、DRIC约4.63GiB，仅为有界步骤测量，不是长期峰值保证。
- 16行RUNNING、20行QUEUED，8卡各2个计算PID；16个工作进程PID/CWD/argv均匹配，连续两次读回的接受步全部增长，异常扫描0项。
- 16份实际resolved配置与发布矩阵逐项一致；scratch、无resume/旧checkpoint、E200保持不变。配置中的`launch=false`是准备入口的防自动执行元数据；本次逐行命令显式带`--execute`，实际运行证据以PID/接受步/运行状态为准。
- `release_remote_readback.json`保存完整时点证据，`release_verification.json`保存核验摘要。所有任务仍在预热阶段，尚不能声明E21以后机制持续生效或最终效果；后续90份新配置和依赖项未启动。没有自动目标评分。
