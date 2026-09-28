# N607首轮六行发布P0/P1审查

审查对象：`20260928-phase1-ir-eg-source-screen-s392005-r01`，仅检查下一次真实启动的直接正确性。未重审未改算法，未修改代码，未执行远端命令或训练。实际远端GPU空闲和文件存在性由发布owner的preflight及launcher核实。

## 首次审查：发现1项P1（已由下方定点复审闭合）

**P1：PyTorch2.1不支持IR激活路径的autocast查询签名。**位置：`code/cvsrffi/xuc_fusion/ir_solver.py:159`。

该行在正常FP32环境中执行`torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')`。N607使用torch2.1.0+cu121，该版本的`is_autocast_enabled`不接受参数；第一项为False后，第二项抛出TypeError。因此IR、IR_G0、OR_EG、IR_ENCODER_OFF会在E21第一次进入响应分支时确定性失败。E1至E20及只执行模型前向的scratch checkpoint smoke不能覆盖此错误。

官方v2.1.0声明分别为无参`is_autocast_enabled()`和`is_autocast_cpu_enabled()`，见[PyTorch2.1.0源码](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/_C/__init__.pyi.in#L1103-L1107)。同包`code/cvsrffi/game_tracking/solvers.py`已通过捕获TypeError回退到`is_autocast_cpu_enabled()`兼容旧版本。IR入口应采用同一兼容语义，并用既定远端相关测试覆盖至少一次active IR主步。此项属于环境导致无法完成本次训练的直接错误，不是新增实验门槛。

## 已核实的启动约束

- spec恰为SIM、EG、IR、IR_G0、OR_EG、IR_ENCODER_OFF六行，model seed固定392005；没有BR或额外seed。六个实际row配置均通过只读`validate_prepared_row`规范化检查。
- 六行均为200epoch×222接受步＝44400步，采用`reuse_origin_used_scales`。六个输出路径互异，命令没有resume参数；登记的checkpoint来源为空、初始化为scratch、query权限为none。
- GPU映射为1/3/4/5/6/7，每卡一行。launcher按GPU UUID检查所分配设备当前无compute进程，不修改GPU0/2现有任务，也没有kill、重试或依赖自动启动路径。
- launcher拒绝已存在run/log根目录，独占创建本次根目录及日志，先运行scratch smoke再提交六行；保存每行PID/command/CWD/GPU/output/log。其`SUBMITTED`状态不冒充独立核实后的RUNNING。
- scratch smoke只从既有source contract声明的L_s物理ID选取最多16条记录，保持原source RX/day及domain映射；从零构造模型，保存并CPU反序列化其本次初始化checkpoint，严格恢复并比较有限前向输出。该checkpoint不会作为任何训练行的初始化来源。
- source合同路径沿用既有`phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json`。训练启动器要求该路径，正式数据构造后比对已有物理角色ID；本审查没有读取目标样本或要求数据重验。

## 证据边界

以上为本地源码、配置及只读配置解析审查。发布owner仍按既定流程完成一次归档校验/编译、远端相关兼容测试、scratch smoke及启动后独立进程和日志读回；这些步骤的结果未由本报告预先声明。未发现其他P0/P1启动缺陷。

## 原问题定点复审：PASS

2026-09-28仅复核`ir_solver.py:159–165`本次兼容性diff和新增`tests/test_ir_runtime_contract.py`，没有扩大审查或重跑测试。

修复先读取无参CUDA autocast状态，再尝试现代CPU查询；遇到PyTorch2.1的TypeError时改用该版本支持的`is_autocast_cpu_enabled()`。正常FP32可继续进入IR主步，CPU或CUDA autocast仍被拒绝，因此没有为兼容性放松FP32边界。原P1已在源码层面闭合，目前本次六行候选没有未关闭的P0/P1审查发现。

新增测试模拟旧查询签名，同时保留真实CPU autocast状态和完整IR主步：FP32分支要求接受且只提交一次；CPU autocast分支要求拒绝、模型恢复、optimizer无状态、solver及外部commit计数均为0。集成owner报告该文件本地2项通过；审查者已核对测试内容，没有重复执行。真实N607 torch2.1 CUDA相关测试及一次scratch smoke按既定发布流程继续，其结果仍须由实际远端证据确认，本结论不预先替代这些结果。
