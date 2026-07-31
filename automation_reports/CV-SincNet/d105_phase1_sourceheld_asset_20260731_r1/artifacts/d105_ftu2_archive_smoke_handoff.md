# D105-FTU2精确archive独立验证交接

状态：`PASS / NOT_N607_AUTHORIZATION / NO_PERFORMANCE_RESULT`

|项目|结果|
|---|---|
|source commit|`2d948ce981b9008522f825cfe6d868bce08cb624`|
|archive|SHA256=`e58240a0a358893c0c90ce0b3cb9c202eed9e6907272fa0d587d160f3fb8ec23`；242964480B；4770成员|
|安全门|单一`source/`根；绝对/逃逸路径、重复、链接和特殊成员均为0|
|runtime/method|`8797de12f035db609aeb6f453f096571f216d0d514d6705344e763f5ec63a498`/`9a87e51de4d775ff2ea05e59654afaa62844edaf2def942d8f73c8e289ea61e6`|
|54文件|Git blob=tar=解包=manifest 54/54；LF54/54；独立pyc54/54；源码零污染|
|CLI/测试|9/9正式help；固定10文件238/238通过；非256早拒绝通过|
|固定批合同|8400行=33次forward；末批208实＋48零填充；每次实际shape256|
|真实checkpoint|1/208/256行实际export相对独立256零填充reference三路max_abs=0；state不变；GRB未导入|

外部总验证JSON SHA256=`16d4399a414383a5b889a4a7e030a8098b2c5f7f03cdf1136a215f1d075bba62`，完整handoff SHA256=`565b14707d622591deb070d074238458ca4be25a6cb1e173497d665dccd09baf`。本机没有8400行真实source IQ，完整reference parity必须保留为新N607第一次tap硬门；本交接不授权N607或性能声明。
