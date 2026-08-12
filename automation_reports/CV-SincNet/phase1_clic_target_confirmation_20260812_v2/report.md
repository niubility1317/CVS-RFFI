# Phase1 CLIC目标确认缓存v2预注册报告

## 状态与唯一修复

- 实验ID：`phase1_clic_target_confirmation_20260812_v2`。
- 当前状态：`LOCAL_VERIFIED / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- v1已不可变封存为入口技术失败：formal launcher调用1次、exit=3、builder PID=0、工件=0。根因是外层重定向先创建了受保护run/log根，launcher正确拒绝覆盖；不是代码、数据或性能故障。
- v2唯一修复是新run/log/output ID，并规定outer输出只能写到run/log根之外。方法、数据、TX/RX/day、样本数、场景、seed、builder及loader字节均不变。

## 冻结配置

- scope=`phase1_clic_target_confirmation`；roles=`target_registered_known,target_unknown`；receiver=`20-1`；days=`0,1,2`。
- registered TX为`14-10,14-7,20-15,20-19,6-15,8-20`；unknown TX为冻结20类清单。
- 每TX=120个物理样本，每scene=40；总3120，每scene1040，其中registered240、unknown800；三scene physical ID两两不交。
- dataset seed=`713101`；scene seeds=`7131010,7131011,7131012`；ManyTx底层数据不变。
- 输出=`runs/phase1_clic_target_confirmation_20260812_v2`；日志=`logs/phase1_clic_target_confirmation_20260812_v2`；outer=`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_target_confirmation_20260812_v2_outer.out`，不得预建run/log根。

## 发布门与运行合同

- 缓存生产实现已独立终审`ALLOW_CACHE_STAGE，P0=0，P1=0`；builder全量13/13、postfreeze132/132、Phase1核心190/190通过。
- v2本地必须通过JSON解析、builder/loader编译、launcher`bash -n`及dry-run精确1行。
- N607仅普通账号、GPU0、唯一formal launch=1、retry=`NO`；启动前run/log/release路径必须不存在。
- 成功要求3个NPZ和cache_set manifest由production loader重开：总3120、每scene1040、registered240、unknown800、每TX×scene40、finite、跨scene物理ID不交、零clean/query/truth/fit/update/selection访问。
- 成功仅记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`，不读取或报告性能。

## 待N607回填

- Git commit、archive/SCP/release、静态门、唯一launch/PID/GPU、工件SHA/count/schema及SSH清理证据。
