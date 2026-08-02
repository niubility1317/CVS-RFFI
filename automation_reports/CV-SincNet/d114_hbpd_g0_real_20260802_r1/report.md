# D114-HBPD-qKNN真实G0报告

状态：`LOCAL_VERIFIED / REAL_ARCHIVE_G0_FUNCTIONAL_PASS / G1_ENTRY_ALLOWED / NO_PERFORMANCE_RESULT`

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
|Git|理论commit`f77c34ec`；实现尚未提交；本地功能筛选后按裁决提交，不push|

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

- 方法提交：`a6ec35a2940ef686e6e65145d95ad5beadb2c3b0`；工作树在打包前clean；不push。
- 首次commit-bound源码包：`source_a6ec35a2.zip`，SHA256=`9f4dc68402ae060ef03fcacc5f482304d006d62f38d5336e82010d454fc99b78`，56,754,279B。报告预登记提交后允许另建最终包，但不得改代码或复用同名包。
- 输入同步到`<run>/input/d106_ls_strict_tap.npz`与`<run>/input/d106_ls_strict_tap.receipt.json`；各自来源为本报告第2节固定路径与receipt，tap SHA必须保持`48b92f...afa2f`。
- Python固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；先在`<run>/source/code`执行D114三个实现文件的`py_compile`和聚焦测试，不运行其他矩阵。
- 冻结child command：`python code/scripts/run_d114_hbpd_g0_one_shot.py --archive <run>/input/d106_ls_strict_tap.npz --receipt <run>/input/d106_ls_strict_tap.receipt.json --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d114_hbpd_g0_real_a6ec35a2_20260802_r1 --output <run>/artifacts/result.json`。
- 唯一Terra Max runner负责direct preflight、资源记录、精确同步、远端SHA/解包/compile/test、不可覆盖根检查、唯一detach、PID/CWD/cmdline/GPU与日志增长核验、完整日志读取、artifact回收和SSH清理；主agent不并发启动。
- 预期artifact只含无truth功能证据。成功要求行数588、fold数28、K=`1/5/10`、query fit/update=0、truth scoring=false、三Kargmax变化均非零且远端execution root与本地一致；否则保留证据并停止，不重启、不调参。
