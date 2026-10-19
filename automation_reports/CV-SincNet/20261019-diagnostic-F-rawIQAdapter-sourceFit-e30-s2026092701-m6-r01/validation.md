# 本地验证记录

- 24 个针对性协议负例通过，覆盖 physical IDs、原始处理、末次 origin、seed/config、复用 NPZ 形状与类别顺序、教师只读和完整日志流。
- 三个新入口及两个嵌入远端脚本可编译；原 GPU4/6/7、总进程上限2、free12GB及共享锁约束保留。
- 本地只读四组全部4440条L/U记录，采样坐标完全相同；未读取真实checkpoint或IQ，未执行NN或远端操作。
- 唯一一次独立 P0/P1 审核 PASS；无直接缺陷或定向复审。审查覆盖旧缓存接口、L权限、默认头、4321NN预算和发布路径；未执行真实CP/IQ/NN或远端。
- 待实际执行：Git代码身份、单包传输/远端compile/preflight、launcher-first真实CPnoquery smoke及独立post-state。按产物更新，不新增审批或重复审核。


实际执行：VERIFIED。代码 d94bb9c949bdf53c96efa8acc16c1695fcb44f21，发布 069bb52daf5547cfc22fb8b83c311385d93011fc；单 archive SHA 比较、N607 preflight、远端compile成功。PID2363317 / GPU4；首个独立快照证实PID/CWD/argv，随后任务快速完成，未捕获实时NVIDIA进程，不能声称该项已直接观察。独立产物证实首步真实NOAS FINAL30 smokePASS并继续、四模型4321真实源包前向、900信道包、全部注册状态不变、0optimizer/EMA/headfit/V/target。完整源产物已收回，不读取CP/IQ/formal。

登记修正：将多模型共用目录误写成各row唯一output_root曾导致索引漏项；已改为真实四个readout文件及TC/TS原复用目录，validate通过并恢复索引。仅修正登记描述，未修改或重跑任何实验产物。
