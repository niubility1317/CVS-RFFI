# D105 Phase1 R3 N607预启动失败交接

## 结论

- run ID：`d105_phase1_sourceheld_2eaa1b11_20260731_r3`
- 终态：`LANDED_PRELAUNCH_HASH_MISMATCH / NO_PERFORMANCE_RESULT`
- 未启动pipeline、未产生strict tap/source-held预测或分数、未构建component、未执行authority seal、未启动Target25。
- 此run root已保留，不得覆盖、恢复、复用或retry。后续修复必须重新冻结runtime、提交并使用新的run ID。

## 已完成的受控落地

|项目|结果|
|---|---|
|冻结Git提交|`bf9406429cb5540e785bc3e61434c682ab548bb9`|
|runtime/method identity|`48ce446cb406aad67902c80547a48ffbd95d496c728725f2d99fc51b3433f9da`/`cdae572cad22721351620828cd1ec36ae1d3432d4b04c70a4c60359a64339d2a`|
|本地Git archive|4745条目、243425280B、SHA256=`7351007ff817ed1dc82fce09ada195ec94c1bdc98ffb5ab336cb56c155bb5bca`；单一`source/`根；仅常规文件/目录|
|远端输入校验|archive、launcher和两份D102公开撤销文件SHA均匹配；4个input已冻结为0444|
|远端解包|4745条目解包至`<run-root>/source`；无符号链接|
|GPU/资源|预检及启动前复检均为8×RTX3090无compute进程，GPU0允许此run；磁盘可用7.4T|
|SSH清理|每次SSH/SCP后均确认无`ssh.exe`，无到N607/bridge:22的ESTABLISHED连接|

远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_2eaa1b11_20260731_r3`

## 预启动权限位假阴性

第一次预启动验证未触及source或pipeline。它在将4个已验证input冻结为0444后失败，原因是runner检查把`stat -c %a`返回的`444`按十进制参与位运算：

```bash
# 原表达式：假阴性
[ $(( $(stat -c %a "$f") & 0222 )) -eq 0 ]

# 经主控明确授权后的修正表达式：精确接受预期0444
[ "$(stat -c %a "$f")" = '444' ]
```

修正只发生在一次性预启动验证脚本内，不改source、输入、launcher、方法、配置、run ID或启动命令。复检确认`pipeline_stage1.pid`、`pipeline_stage1.log`和`pipeline_stage1.exit`全部不存在，GPU仍空闲后才继续闭包验证。

## P0：runtime闭包与Git archive字节不一致

修正后的检查中，`bash -n`通过；canonical runtime loader在54项闭包的第一阶段抛出：

```text
D105Phase1BundleError: D105 candidate runtime core file SHA256 drift
```

它发生在py_compile和checkpoint加载之前。`verification/pyc/`为空、`prelaunch_verification.json`不存在、source中`__pycache__`数为0。因此未获得PyTorch版本或实际checkpoint加载策略，也没有任何性能或Phase1资格结果。

下表是冻结manifest的期望SHA与从远端`source/`直接回收副本的实际SHA。24/54项不一致；回收副本位于本目录的`remote_runtime_mismatches/`。

|runtime相对路径|大小(B)|期望SHA256|远端实际SHA256|
|---|---:|---|---|
|`cvsrffi/phase1_rb_metabias4_bundle.py`|38004|`a88fafc8c948e2ecfe223baa9f84012f831d88b8423f5a3c5c5e65d80db3fb06`|`7e2eb67a592a94de8be1091c29b6df796d8122ffe76b3a7424e985d694ae8c5f`|
|`cvsrffi/rxid_metabias4_bundle.py`|29777|`817bea937e2f5bcdf45f9a5a7db2a5c68c656666e61c30ae39bedbe99f372414`|`0236ba3895dded5f37a06c68fcb4f6f306e93f095a15ffa923b98ec1ce819f3c`|
|`cvsrffi/rxid_metabias4_held_execution.py`|30788|`571ddb448cd44131a05ff6187fbb66ad20ae115af57412447e8a92c08c39cc1e`|`45223e824408564a2cfd12a4a9b0be901d17c3680bc119b944083b91eb98d710`|
|`cvsrffi/rxid_metabias4_phase1_trainer.py`|66288|`e00ee30081efcfd42917c97bbe3997958fa9dfdc67cb45194f420edca4d00527`|`7ada8b7b0e33be8f2d45b211f6042fb70c713b11bc39b73361bd8abcbae67b45`|
|`cvsrffi/rxid_metabias4_source_archive.py`|21945|`dcfe6f0c8d0c49b06d7482185329a389eba3f14f542790d6d3577d8b48f3e764`|`26ba06ffab89cc592919fba6e5b99c046971f565f6e0df954641f8d2f40619f5`|
|`cvsrffi/stage2_d105_cbrc.py`|55256|`9f07c1db63bf518ce11089bd099462cf22d19c2c2c643731f80a5faef53203f6`|`be18be5297dc9285227341b6c68887160ada5c0778714524865cf90de90ca048`|
|`cvsrffi/stage2_d105_feature_tap.py`|7914|`b0f5045f97e8af761a537518c546969ae651412c38fa4dfdeec5b712e86698ca`|`ce31113fb9f64f54c0109cc458b62378b0848204adc2634ba1c7218719efb8bf`|
|`cvsrffi/stage2_d105_four_arm.py`|13941|`0e564aee2a68dae6192419eb02def5e5d1b062438dece21e7cd3a405262cad15`|`6f001c4960e0292987f72ed9c164f1ab3111e1f4ed3af275a587d0504c994c52`|
|`cvsrffi/stage2_d105_phase1_authority.py`|39982|`fe81a728bcd8e1047a40069b9d9954aed2af1c89b98633489ccf2b922b4364bd`|`e9b4f083d7b1dc381aa424040d3f99d8b4fb8d9820c6a412fe52cc43b7471b7d`|
|`cvsrffi/stage2_d105_phase1_bundle.py`|192173|`7ee1af4b75aa3f6582185a62e9c68a6a077e51d98db0845d30d4a42b0fc40e6f`|`3620192ca9f6694eae9b002a596ac675b3e6dbe1ecb2c4e21b933da811c656c7`|
|`cvsrffi/stage2_d105_query_evaluation.py`|66930|`a04305cdfe0c6309f419b95511c08c628f703e0ce6fb782dd2baacd6e6d7d7fd`|`f9c669d8d45a10f83d20d5b8f8c9aac27d03e9f03f96631f5f0f98396a5ab916`|
|`cvsrffi/stage2_d105_target25_inputs.py`|44660|`c15831b86d03c8f5b7c3528a7e18460859349cec7ac233cd33da4db9cab97c71`|`a6a736c9b9b3a6e69364d9d9f69b87c42cf6885d26a3be04d27050928d6519e1`|
|`cvsrffi/stage2_d105_target25_launcher.py`|30256|`17a1ec263a76f15263d0d68ccfe65ee281fe5363e4a6147a8439f3b0d0a8e703`|`65ac3884162432d9c29c6d54b26b866c959e8e51aabffe5e12e7c6bb16041b63`|
|`cvsrffi/stage2_d105_target25_runner.py`|155458|`540a2b22af48da96afdec8911a2105819f3d7c2e2b61a2d189358d6a3e9df73b`|`5c9a56681ff012f817e0d299ee1759d533555e113729f1d25177b1cba93779cc`|
|`cvsrffi/stage2_grb_jp4_adv_drqknn_bcrr.py`|131122|`a24e0fd6324eb1984c94264de0bd40aa9cf7459845d9a18d5f336c6120b146e6`|`a39128d067337c72dbd47a59fa24ef6f5930171359e4a7d5c092ed13956f33c3`|
|`cvsrffi/stage2_lpo_rc_qknn.py`|50231|`e88a55c239c31067eb4ae01c729039f73c2a5705d5406a45f761c33d97865492`|`54072e9c2cd8d05752b6ca264753df2aed9f016ff4cfe3fa27bb6b090cf0be75`|
|`cvsrffi/stage2_rb_metabias4_qknn.py`|51552|`fd051f6c13a3bec243fa9ffed3c5841becd3ba2b5dc17a9896a011764389a3fd`|`f76b570706bf511e82ad56d39458240b53ab18212e4b68eb39e45f6b7a7aa74a`|
|`cvsrffi/stage2_rxid_metabias4.py`|30377|`1750dd1aea775034f74e921ca4f44b12df20c60ccb82182d97dcc195ded35294`|`c9b834e9b4ffe5df30e067711d66ca40e2427fa0758650d717dbb2b2f08e3541`|
|`model_dual_cvsincnet.py`|31251|`f81bb37b4cb1c06a4a291e0ea664784a4ee1410c6b135bf68fd9ed72ff163047`|`92239cc0980714d0a510a07b39af555077622738b2379fd0575d1b78b7206afd`|
|`scripts/build_d105_phase1_bundle.py`|25971|`376aface23de05d9cb9cce86d1526a2ee788ccaad0af482b80b97ec6d1883220`|`f3551ed06b5d7fdc2e7a7d3df6fd0a6f086b30839a94c382f70adb73893fc26f`|
|`scripts/prepare_d105_target25_inputs.py`|1498|`9d31f9d52c64cc7f5b43e5576c07ce14d3be99b2a7f028a89b38eab321e7c882`|`7f9d4dad8488422f58a3a6fdd36a7c72ef6881e78d3c043398f20b1adcebdb49`|
|`scripts/run_d105_four_arm_real_feature_smoke.py`|12415|`d7983c09a3ee80592ce9469b27ec8c10a1737bfda71a3db79c840e9245497173`|`ad023f5003584faf99ea73ef1dcdda5aa1397a7184ddf4e2d97a47a284c3f015`|
|`scripts/run_d105_target25.py`|10504|`f7a9c3bdafd573a12e52cc5d21f8128ad11b011914cba0de39ea9fc21a2b8f46`|`7a731cc0bceba40b83b9be15ede77b24043f34ad41f6d91eea8dd76ed9372519`|
|`scripts/sign_d105_phase1_authority.py`|19827|`8fa2e0508f2c20136f5a53c2c025aaaff8a755afea510aeec6ed6caf8d3e2fde`|`fdba2bb27ebc164287d72516f623fb5033c5a02dc4cdc130de448e6cd4ca592a`|

## 已回收证据

|本地回收项|SHA256|
|---|---|
|`input/run_d105_phase1_stage1_2eaa1b11.sh`|`6d540868f3347633846ffce92a2bc424e3f63593fc9a0b16333de3f7aad938de`|
|`input/d105_d102_revocation_manifest.json`|`99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734`|
|`input/d105_d102_revocation_manifest.ed25519`|`53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58`|
|`source_configs/d105_candidate_runtime_manifest_20260731.json`|`48ce446cb406aad67902c80547a48ffbd95d496c728725f2d99fc51b3433f9da`|
|`source_configs/d105_candidate_method_lock_20260731.json`|`cdae572cad22721351620828cd1ec36ae1d3432d4b04c70a4c60359a64339d2a`|

只复制了公开launcher、公开D102撤销manifest/signature、两份冻结配置及24个明确不匹配的runtime文件。未访问、搜索、复制或输出任何private PEM；未复制数据集、checkpoint或source cache。

## 保留的远端状态

```text
<run-root>/input/
  d105_phase1_source_bf940642.tar
  run_d105_phase1_stage1_2eaa1b11.sh
  d105_d102_revocation_manifest.json
  d105_d102_revocation_manifest.ed25519
<run-root>/source/
<run-root>/logs/                 # 无pipeline_stage1.pid/log/exit
<run-root>/verification/pyc/     # 0个编译文件；无verification receipt
```

下一步只应在本地确认manifest字节生成规则与Git commit blob/archive字节一致，重新冻结并独立复审后，以新run ID重新发布；本run不具有任何可分析性能结果。
