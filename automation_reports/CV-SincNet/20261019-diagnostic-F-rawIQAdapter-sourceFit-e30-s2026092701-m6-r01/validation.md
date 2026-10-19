# 本地验证记录

- 24 个针对性协议负例通过，覆盖 physical IDs、原始处理、末次 origin、seed/config、复用 NPZ 形状与类别顺序、教师只读和完整日志流。
- 三个新入口及两个嵌入远端脚本可编译；原 GPU4/6/7、总进程上限2、free12GB及共享锁约束保留。
- 本地只读四组全部4440条L/U记录，采样坐标完全相同；未读取真实checkpoint或IQ，未执行NN或远端操作。
- 唯一一次独立 P0/P1 审核 PASS；无直接缺陷或定向复审。审查覆盖旧缓存接口、L权限、默认头、4321NN预算和发布路径；未执行真实CP/IQ/NN或远端。
- 待实际执行：Git代码身份、单包传输/远端compile/preflight、launcher-first真实CPnoquery smoke及独立post-state。按产物更新，不新增审批或重复审核。
