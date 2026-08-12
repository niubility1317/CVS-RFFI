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

## N607运行与验收封存（2026-08-12）

- 冻结Git commit：`68e5d00af454000fdba50580a0d6edfb6873c0c2`。归档由该commit直接生成，未解包本地、未改动dirty树；archive bytes=`267489280`，SHA256=`C2F4C170E019AF4453D52FDF5CB65D3715397AA43BD6556A32B5DE0653C24A13`。SCP恰1次到项目根，远端bytes/SHA闭合。
- 远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_target_confirmation_20260812_v2_68e5d00a`；stage目录解包后原子改名。四文件静态SHA闭合：spec=`d1b20cd9bff5d33646a38687d7e62ac1cdf19ef137a0020f7ccefd17d057c361`；launcher=`08294d0ded02b90fdfe367338d54c5f9779a47532aeb42bfff37c187734c1f8c`；builder=`bcd6b0d1dd784ae518d26c1889645d2dd4b22bbcdee4158fea3dc606404370f2`；production loader=`dceb7b6e32ba63ede0cc91dbf839779976a4ac31bd71b1e6f2a187cf76ef481e`。ManyTx存在；JSON解析、builder/loader`py_compile`、builder`--help`、launcher`bash -n`和`bash <launcher> --dry-run`均通过。归档不保留launcher可执行位，直接执行会返回126；正式入口始终使用`bash`，未chmod、未改release。
- 唯一正式命令：`nohup bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_target_confirmation_20260812_v2_68e5d00a/code/scripts/launch_phase1_clic_target_cache_v2_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_target_confirmation_20260812_v2_outer.out 2>&1 &`；formal invocation=`1`，retry=`NO`，outer PID=`2692275`，builder PID=`2692287`。outer文件位于项目根、run/log根之外且0B；run/log由launcher内部创建，未被外层预建。两进程均正常退出，无匹配旧runner或其他run进程。
- 最终状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。工件同一run且不可覆盖：

|场景|NPZ bytes|NPZ SHA256|rows|registered|unknown|
|---|---:|---|---:|---:|---:|
|`leo_clear_weak`|3641482|`3217da7846e4ab2bdb46ab4ec61604c4b5a4e30c544a2462865ffca2f7eaa530`|1040|240|800|
|`leo_low_elev_weak`|3654054|`3f8160557e91c7f8ede9d460bbc342cc5cf29660ddded0293630967a550eb0a4`|1040|240|800|
|`leo_rain_weak`|3637310|`998c0172b9eb9cb3c52f1fa9583999f4e32b320bba5caa9f1438997c31133a51`|1040|240|800|
|`cache_set.json`|6248|`47c4ca2acadd992b99ddf007dc60ff439d63d859ed5e160bd5ee700dfcef3b11`|—|—|—|

- production loader`load_verified_leo_weak_cache_set`重开通过：schema=`cvs_leo_weak_iq_cache_set_v2`，scope正确；总physical observations=`3120`、unique physical samples=`3120`；每场景1040；每场景角色精确为registered-known=240、unknown=800；所有数值数组finite；三场景physical ID交集为0；manifest与NPZ SHA均闭合；`clean_sample_access=false`、`phase2_single_observation_compliant=true`，成员清单先于IQ读取的禁用访问检查通过，未发生clean/query-truth/fit/update/selection访问。
- 运行健康与清理：target log=`4661`B、PID表=`168`B，无`Traceback`、`RuntimeError`、OOM、NaN、权限或失败指纹；GPU0—7均`0%`利用率、约`1MiB`显存；远端无run-owned进程；本地`ssh/scp`进程为0且到N607:22连接为0。未读取任何性能指标，v2是独立fresh run，不是v1 retry。
