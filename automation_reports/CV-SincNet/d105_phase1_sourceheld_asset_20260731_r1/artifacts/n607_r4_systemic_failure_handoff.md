# D105 Phase1 R4 N607运行交接

## 终态

STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT

- run ID：d105_phase1_sourceheld_d23469ba_20260731_r4
- 远端run root：/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_d23469ba_20260731_r4
- 唯一detach已执行1次，PID=2726125，fresh-run retry=NO。
- 当前无run-owned进程；8张GPU均为0%/1MiB；本地ssh.exe为0，N607/bridge:22无ESTABLISHED连接。
- 未访问Target、未启动Target25、未执行authority签名或seal，且没有任何性能结果。

## 冻结身份与本地门

|对象|核验结果|
|---|---|
|source archive commit|d23469ba54afe00c284aa9b78b025def2b22fc43，严格由git archive --format=tar --prefix=source/生成|
|prereg/bind/correction|03e3ff67003f16b6c39596a521f5bfdf0401850c/23e03647afaabc49532206234fcdfde732ed5b22/79671132fb7df1948b851d0af16870d3a36e2340均已核验|
|archive|242810880B；SHA256=d0772371c4ec378fd090bc1f4f7e6a974f2e35890273e309a433c94c1525c9dd；4750成员=4183个regular+567个目录；单一source/根；绝对/..路径、链接、重复项均为0|
|runtime/method|dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc/ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030；以source/code为manifest规范根完成54/54逐字节核验|
|launcher|5624B、LF-only、bash -n通过；SHA256=b72fef60ab14ec86e9b53cc3355d07ac50598a3d6788000b4962726c89271d35|
|D102公开输入|manifest/signature SHA256=99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734/53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58|
|R5 receipt|65f8f211c01b8b72b4f4d7a385d9c1747b16dae9f14bddd32457ccf2f402c822；LOCAL_RELEASE_GO/P0=0/P1=0/P2=2|

首次本地54项核验曾错误地把manifest相对路径根定位为source/，因而记录为54项缺失；该尝试未改变任何源字节。随后按清单规范根source/code重做，54/54通过。两个记录均保留。

## N607落地与prelaunch

- 直连普通N607账户preflight通过；R4根落地前不存在；历史R3根d105_phase1_sourceheld_2eaa1b11_20260731_r3存在且未触碰。
- 创建新的非覆盖run root及最小input/、logs/。仅同步source tar、D102 manifest、D102 signature和冻结launcher；未复制私钥。
- 四个input逐项SHA和大小通过后冻结为0444，所有权限判断均以stat -c %a返回的字符串444比较。
- archive安全解包至source/，并生成只读verification/archive_extract_receipt.json。
- prelaunch通过：canonical runtime/method loader、54个独立pyc、远端launcher LF/bash -n、checkpoint SHA和受限加载策略。
- 实际远端PyTorch=2.1.0+cu121；safe_globals不可用，因此策略为legacy_pickle_exact_frozen_sha_only，仅在精确checkpoint SHA匹配后执行；checkpoint含195个state tensor。

## 技术失败与产物闭合

detach后首个tap-cache阶段在strict tap产物之前退出。完整日志的归一化异常指纹为：

TypeError: expected np.ndarray (got numpy.ndarray)

调用位置为stage2_d105_phase1_bundle.py:1658的torch.from_numpy(batch)。pipeline_stage1.exit为1，日志为992B，SHA256=9aa65a3e034283ee481b0bb12792ac6fe8fd6e51fb3b71470abe259de49ffe10。

|闭包项|结果|
|---|---|
|strict tap archive/receipt|均不存在|
|prediction manifest|不存在|
|truth-open receipt|不存在|
|score artifact|不存在|
|gate receipt|不存在|
|component|0个文件|
|output regular files|0|
|运行后54项runtime核验|54/54一致，0项漂移|
|source pycache|仅运行时生成code/__pycache__和code/cvsrffi/__pycache__两个目录；源文件SHA未变|

这是launcher-wide、零预测前的确定性技术故障。该run不得重启、修补、覆盖或复用，亦不得作为性能或formal asset证据。

## 已回收文件

|本地文件|SHA256或说明|
|---|---|
|source_d23469ba54afe00c284aa9b78b025def2b22fc43.tar|d0772371c4ec378fd090bc1f4f7e6a974f2e35890273e309a433c94c1525c9dd|
|archive_integrity_local_canonical_code_root.json|本地archive结构与54/54规范根核验通过|
|local_frozen_input_integrity_v2.json|launcher/D102/method/R5本地冻结门通过|
|remote_recovery/pipeline_stage1.log|9aa65a3e034283ee481b0bb12792ac6fe8fd6e51fb3b71470abe259de49ffe10|
|remote_recovery/pipeline_stage1.exit|1；SHA256=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865|
|remote_recovery/archive_extract_receipt.json|f7f19c31a4e71f6a3439b2aafe7ed494d911317dc8e47b88d047043733485c59|
|remote_recovery/prelaunch_verification.json|9ae017224273cb665033cba08088525c892fd7e6ac25e0efe33d6241739000c4|
|remote_recovery/post_failure_inventory.json|7dadbe642f17d96e747522bdf42ff45420ae4dde06f61def30ba0c756817951a|
|remote_retrieval_integrity.json|回收文件、prelaunch和终态闭合的本地复核通过|

## 后续边界

需在本地修复该NumPy/PyTorch对象边界问题、独立复审、重新提交并预登记新的非覆盖run ID后，才能再次申请release。R4本身仅保留证据；不得重启或改写。
