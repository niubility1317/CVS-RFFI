# D105-FTU1实现与本地验证

状态：`LOCAL_VERIFIED / R10_GO / NOT_N607_AUTHORIZATION / NO_PERFORMANCE_RESULT`

## 1.功能修复

R5根因是Phase1把D105双backbone模型接入GRB旧strict tap，旧路径只执行`id_backbone`，导致`z_dom=None`。D105-FTU1将Phase1正式`_strict_forward`唯一接到`stage2_d105_feature_tap.extract_d105_feature_tap`：

- `z_id`来自`id_backbone.feat_joint`；
- `pre_relu`来自同一次`joint_proj.0`hook；
- `z_dom`来自同一received IQ的`dom_backbone.feat_imp→dom_enhancer`；
- 不存在GRB旧tap回退或identity-only fallback；
- export入口分别检查hook标志、三个字段的float32/`[N,160]`/finite、ReLU绑定和execution path。

## 2.入口级回归

新增的checkpoint形状真实CVSincNet测试直接执行Phase1 `_strict_forward`和一行IQ的`export_d105_phase1_strict_tap`正向闭环。旧GRB helper被替换为必失败函数后，新入口仍成功，并验证：

|项目|结果|
|---|---|
|id backbone调用|1次|
|dom backbone调用|1次|
|dom enhancer调用|1次|
|same-IQ|enhancer第二输入与原received IQ精确相等|
|域特征来源|enhancer第一输入与`dom_backbone["feat_imp"]`精确相等|
|严格输出|`z_dom=float32[1,160]`；`z_id=ReLU(pre_relu)`|
|正式export|receipt写`eager_forward_hook`，archive内`z_dom/pre_relu`与直接tap一致|

另有7项export字段破坏负测，均在创建输出目录前fail-closed。

## 3.主agent真实checkpoint复验

使用SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`的同代生产checkpoint，在`ssr-gpu`/CPU上安全加载195个state tensor。D105-FTU1得到：

```text
execution_path=eager_forward_hook
z_id=[2,160]
pre_relu=[2,160]
z_dom=[2,160]
hook_exact_bytes=true
relu_parity=true
finite=true
```

fresh进程调用前后`cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr`均未进入`sys.modules`。完整机器记录见`analysis/d105_ftu1_real_checkpoint_validation_20260731.json`。

## 4.回归、身份与独立审查

|门|结果|
|---|---|
|10个D105/LPO-RC测试文件|223/223通过|
|54文件runtime闭包|54/54哈希核验通过|
|candidate runtime|`873879aad707fd2407b7645de45daa68fec1d3537feaf9fd57fe98b3ab059214`|
|candidate method lock|`7d33662750b160fce82217dace9e1933aa8e43ea2a0df19f59e28adcf8bb4848`|
|Phase1 bundle代码|`9060e1f8dc65e24ce2b3843c098f70b2f2837950fe4314025e5076f183af95e4`|
|R10独立审查|`LOCAL_CODE_REVIEW_GO / P0=0 / P1=0 / P2=2`|
|R10收据|SHA256=`31ebec822064b4db7a3e5f4d419ee0ce8c4a493bb454ef1c0629c265164b8831`|

两个P2是：低层loader仍保留历史execution-path字符串兼容；提交后仍需从精确Git archive执行SHA钉定真实checkpoint source-only no-truth smoke。两者都不授权N607或性能解释。

## 5.证据边界

本次只修复Phase1技术出口，不改变CBRC、LPO-RC、qKNN、Target25的25job矩阵、seed、性能门或协议权限。没有Target访问、query truth、role/quota、跨query重排或性能计算。D105仍没有Phase1 formal asset和Target性能结果。
