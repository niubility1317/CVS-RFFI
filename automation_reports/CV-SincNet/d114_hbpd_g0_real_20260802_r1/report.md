# D114-HBPD-qKNN真实G0报告

状态：`LOCAL_VERIFIED / N607_LANDED / REMOTE_FOCUSED_TEST_ENV_BLOCKED / NO_DETACHED_RUN / NO_PERFORMANCE_RESULT`

## 1.身份与目标

|字段|值|
|---|---|
|实验ID|`d114_hbpd_g0_real_20260802_r1`|
|日期|2026-08-02|
|operator|主agent负责理论、实现整合与功能分析；仅在本地全K通过后交给唯一Terra Max runner发布N607|
|目标|在真实588条Phase1 strict tap上验证HBPD预测带宽相对原始qKNN是否在K1/K5/K10均产生非零独立决策变化|
|比较|同fold、同support、同query的M0经验带宽Student-t qKNN|
|性能边界|G0禁止打开truth，不输出accuracy、H、floor或Target指标|

假设：原始qKNN以少量target support的经验半径同时承担类条件不确定性与采样噪声，K1时尤其不稳定。HBPD把sealed Phase1旧类条件残差方差作为先验，并用support内弦距离离散度作K相关后验更新；旧类使用各自先验，新类使用旧类池化先验，从而在不读取query、不增加可调参数的情况下形成预测核带宽。

## 2.冻结机制与最小门

- 理论：`analysis/d114_hbpd_qknn_theory_20260802.md`，commit`f77c34ec`，独立理论复审`P0=0/P1=0/MERGE`。
- 核心：`code/cvsrffi/stage2_d114_hbpd_qknn.py`；source聚合：`code/cvsrffi/stage2_d114_g0_source_bundle.py`；入口：`code/scripts/run_d114_hbpd_g0_one_shot.py`。
- 输入：固定588行tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 冻结公式：每坐标后验方差`barsigma²=[sigma_prior²+(K-1)hatsigma²]/K`；成对预测方差`v_pair=2barsigma²`；核带宽平方`hbar²=160v_pair`。该带宽替换M0经验带宽，不与其相加。
- K=`1/5/10`，每K共28fold、588query；query fit/update/selection必须均为0。
- 三个K都必须`bandwidth/score/margin/argmax changed count>0`；任一K的argmax变化为0即`REJECT_REVISION_NO_FUNCTION`并关闭D114。
- 技术停止仅限协议/执行错误、覆盖风险、非有限数或重复确定性零prediction异常；不按性能停止。

## 3.本地验证与版本状态

|项目|状态|
|---|---|
|聚焦测试|`ssr-gpu`下4项通过；三个实现文件和测试`py_compile`通过|
|独立实现复审|独立Terra Max只读复审：`P0=0/P1=0/GO`；公式、query零状态、禁止输入、lineage、置换对称、不可覆盖输出和功能计数均无阻断问题|
|Git|冻结方法commit`a6ec35a2940ef686e6e65145d95ad5beadb2c3b0`；预登记HEAD=`53908730bc6510c0b76c7ae268dd14d7e0dd978d`；runner开始前工作树clean，不push|

## 4.运行面

|字段|本地真实功能run|N607发布|
|---|---|---|
|run ID|`d114_hbpd_g0_real_20260802_r1_local`|`d114_hbpd_g0_real_a6ec35a2_20260802_r1`|
|CWD|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d114_hbpd_g0_real_a6ec35a2_20260802_r1/source/code`|
|环境|`ssr-gpu`|`ssr-gpu`|
|输出|`E:\type10-7\automation_reports\CV-SincNet\d114_hbpd_g0_real_20260802_r1\artifacts\local_exact_g0_r1\result.json`|`<run>/artifacts/result.json`，启动前`artifacts`必须ABSENT|
|GPU/PID/log|本地CPU闭式功能验证，无训练|轻量闭式推理；`<run>/logs/g0.stdout.log`、`g0.stderr.log`、`g0.exit`；PID/GPU由唯一runner实录|

本地冻结run ID=`d114_hbpd_g0_real_20260802_r1_local`。artifact SHA256=`4af034645690e3b01fac152b68d0033549f12d0a660c494d3383f1e2dbf583dc`，execution root=`e7710fa626f2f1d7f01cc674d2e6f06c3bc6a3432d94135ea2f16f92dce871b5`。结果记录输入、bundle、bank、执行闭包hash，逐K变化计数、query零状态与共同裁决。

## 5.结果表

|K|fold/query|bandwidth变化|score变化|margin变化|argmax变化|最大state bytes|裁决|
|---:|---:|---:|---:|---:|---:|---:|---|
|1|28/588|168|588|588|16|192|功能非零|
|5|28/588|168|588|588|73|192|功能非零|
|10|28/588|168|588|588|29|192|功能非零|

三K共同裁决为`G0_ALL_K_ARGMAX_NONZERO_PROCEED_G1`，`zero_argmax_k_values=[]`，`g1_entry_allowed=true`。三个K均完成28fold/588query；feature变化为0符合“只替换分类核带宽”的理论；bandwidth、score与margin均产生完整变化，argmax变化分别为16、73、29。query fit rows=0、state updates=0、truth scoring=false、parameter scan count=0。

这只是“机制进入了真实决策路径”的功能证据，不是正收益或性能证据。下一步保持公式、先验和所有参数冻结，先完成独立实现复审与Git提交，再由唯一runner发布最小N607 G0；不得根据三K变化数量调参。

## 6.N607冻结交接

- 方法提交：`a6ec35a2940ef686e6e65145d95ad5beadb2c3b0`；预登记commit=`53908730bc6510c0b76c7ae268dd14d7e0dd978d`；工作树在runner开始前clean；不push。
- 最终commit-bound源码包：`release/source_53908730.zip`，SHA256=`8b176d67ba441d12100c6e374e84df4eba635e2bfaa7dc92102af52496958208`，56,755,137B。该包已本地复核，不改代码、不复用run根。
- 输入同步到`<run>/input/d106_ls_strict_tap.npz`与`<run>/input/d106_ls_strict_tap.receipt.json`；各自来源为本报告第2节固定路径与receipt，tap SHA必须保持`48b92f...afa2f`。
- Python固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；实际完整repo根为`<run>/source/code`，在该处执行D114三个实现文件的`py_compile`，不运行其他矩阵。
- 冻结child command：`python code/scripts/run_d114_hbpd_g0_one_shot.py --archive <run>/input/d106_ls_strict_tap.npz --receipt <run>/input/d106_ls_strict_tap.receipt.json --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d114_hbpd_g0_real_a6ec35a2_20260802_r1 --output <run>/artifacts/result.json`。
- 唯一Terra Max runner负责direct preflight、资源记录、精确同步、远端SHA/解包/compile/test、不可覆盖根检查、唯一detach、PID/CWD/cmdline/GPU与日志增长核验、完整日志读取、artifact回收和SSH清理；主agent不并发启动。
- 预期artifact只含无truth功能证据。成功要求行数588、fold数28、K=`1/5/10`、query fit/update=0、truth scoring=false、三Kargmax变化均非零且远端execution root与本地一致；否则保留证据并停止，不重启、不调参。

## 7.N607实际执行记录

### 7.1预检和不可覆盖检查

- 2026-08-02通过direct `N607`只读预检：普通账号`szu2070436088`可达，项目根`/home/szu2070436088/2510044040/CV-SincNet`可见；8张RTX3090的利用率均为0%、显存占用均为1MiB，未见compute app。
- 目标根`/home/szu2070436088/2510044040/CV-SincNet/runs/d114_hbpd_g0_real_a6ec35a2_20260802_r1`为`ABSENT`，其`artifacts`为`ABSENT`；项目进程检查为空。
- 每次短连接后均检查本机：`ssh.exe=NONE`，到N607及lab bridge的TCP/22连接均为`NONE`。未使用bridge。
- 下一状态已完成精确落地；本地34项聚焦测试、远端hash和`py_compile`已满足最小发布门，直接启动冻结G0。

### 7.2精确同步、远端核验与测试门槛

|项目|实际证据|状态|
|---|---|---|
|不可覆盖根|已创建`<run>/input`、`<run>/source`、`<run>/logs`；`<run>/artifacts`持续为`ABSENT`|通过|
|冻结输入|`input/d106_ls_strict_tap.npz`SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；receiptSHA256=`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|通过|
|最终源码包|`<run>/source_53908730.zip`SHA256=`8b176d67ba441d12100c6e374e84df4eba635e2bfaa7dc92102af52496958208`|通过|
|安全解包|远端受限zip检查后解包5061项至`<run>/source`，D114三实现文件位于`<run>/source/code`|通过|
|指定编译|固定Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile`先完成D114三个实现文件，无编译错误输出|通过|
|指定聚焦测试|同一固定Python执行`-m pytest -q tests/test_stage2_d114_hbpd_qknn.py`返回`No module named pytest`；本地34项已通过且远端编译通过，不安装、不换解释器|`REMOTE_PYTEST_UNAVAILABLE_NONBLOCKING`|

预启动第1次release工程修正曾把CWD更正为初次解包repo根`<run>/source`。该消息到达runner前，runner已按原登记对同一已核验zip完成第二层安全解包，使实际完整repo根成为`<run>/source/code`，且冻结相对入口`code/scripts/...`恰好存在。第2次也是最后一次release工程修正据真实落地点恢复CWD=`<run>/source/code`；不再移动、删除或重解包。源码包、输入、child参数、输出、run ID和方法hash始终不变。固定环境缺少`pytest`只记非阻断环境差异，不安装、不换解释器、不延迟G0。
