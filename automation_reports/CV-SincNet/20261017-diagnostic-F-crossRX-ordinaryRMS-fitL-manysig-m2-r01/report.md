# 普通增强输出 RMS 恢复：r01 技术失败

状态 **FAILED_TECHNICAL_VERIFIED**。真实执行 commit be3defc18bb709f48dac0c7c56c1a7c9fa8a8679。没有科学判决，原假设仍未检验。完整原始预登记保存在 [preregistration.md](preregistration.md)。

首模型 seed01 通过真实 checkpoint 的 2 包合成 smoke，并完成 64 包合成普通增强的 canonical 输出/RNG 检查。随后载入既有源缓存并写出合法 fitL5040 元数据，写 resolved_config 时因 native args 中 SatViewStage dataclass 无法转严格 JSON 而失败；尚未执行 raw/ordinary/RMS 源默认头前向，seed02 未启动。根因是日志副本转换只处理 dict/list/tuple/非有限浮点，没有处理原生 dataclass；实际增强日程没有被改变。

实际新增成本：2 合成 NN、1 次 checkpoint 加载、64 合成 ordinary 生成，0 源 NN、0 RMS 视图、0 epoch/更新/教师/另训头/target。原缓存 L6300 与未用 V/历史特征曾反序列化，fitL 元数据5040；U 仅角色 ID，无 IQ/TX 标签。两参考模型的历史 40 model-E/8880 更新/9373356 NN 仍单列，未重新训练。

一次 archive 传输 SHA 比较、远端 compile 与发布器进程读回 VERIFIED；失败后独立读回 root/worker failure、完整日志、smoke 和源元数据。dispatcher 与唯一 worker 已自然退出，无其余实验变更，没有自动重试或覆盖 r01。真实源科学预测为零，因此不能称源身份条件通过或失败。

使用原生 SatViewStage 类定义的纯本地 AST 复现已确认同一序列化失败，0 IQ/checkpoint/NN/更新。下一步仅修复日志副本中的 dataclass 字段表示；如恢复执行，另用 r02 不可变根、保留原矩阵/参数/判据及 r01 成本，不以技术失败调整科学候选。

本轮未提供 A_g、A_s、跨 RX clean 或联合目标的新收益证据。
