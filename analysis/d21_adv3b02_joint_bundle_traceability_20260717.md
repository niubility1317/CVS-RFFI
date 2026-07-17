# D21 ADV3B02联合部署bundle追溯表

## 边界

该模块构建ADV3B02 TorchScript runtime与Phase1 v2压缩原型的不可分割部署包，并产生外置seal/signing request。模块不生成生产私钥或签名；formal loader固定使用项目现有authority issuer、key ID和公钥，仅在真实外部签名envelope通过后返回formal context。

|ID|要求|实现|测试证据|状态|
|---|---|---|---|---|
|JB-01|不得携带raw checkpoint/source路径/sample feature/count/cache spec|固定8成员白名单，不接受`.pth`；JSON key/value与opaque token审计|raw checkpoint、禁止字段与路径式token负例|verified|
|JB-02|standalone v2组件不得自报formal|组件manifest必须保持pending outer seal和`formal_phase2_eligible=false`|正向formal load同时断言standalone false|verified|
|JB-03|runtime、component、class、parity、generation、method共同绑定|outer manifest、content root、detached seal和signature envelope分层绑定|root/member/lock篡改负例|verified|
|JB-04|签名不能由调用方替换|formal API内置固定issuer/key ID/public key及Ed25519 verifier，无可注入verifier参数|错误签名与API surface负例|verified|
|JB-05|无root/signature循环|content root只覆盖8个member descriptor；manifest由seal绑定，seal SHA由envelope签名|signing request与formal load正例|verified|
|JB-06|校验与物化使用同一字节流|包内member及外置seal/envelope均单次打开，同一bytes完成SHA和解析；返回前二次root allowlist|seal替换和unexpected member负例|verified|
|JB-07|class binding必须是有序语义SHA|复用`phase1_tx_class_handle_binding_v1`canonical算法；文件SHA仅由member descriptor绑定|JSON格式重排通过、class顺序改变拒绝|verified|
|JB-08|TorchScript内部不得夹带额外样本状态|复算archive member root、parameter/buffer schema root、state bytes与structure root；parity receipt精确绑定|extra file与unused buffer负例|verified|
|JB-09|真实domain handle可部署且路径不可混入|class/method使用opaque token；domain专用`prefix:suffix`规则支持`rx_day:20`并拒绝Windows/Unix路径|真实domain正例及`C:\...`、`C:/...`负例|verified|
|JB-10|部署读取有内存上限|root member、external JSON与runtime state读取前固定大小门|超限路径由实现断言覆盖|verified|

## 验证

- 联合bundle聚焦测试：10项PASS。
- Phase1 codec/streaming/exporter、joint bundle、Stage2 lifecycle/runner联合回归：61项PASS。
- `py_compile`与`git diff --check`：PASS。
- Windows pytest临时junction清理`PermissionError`为已知退出后噪声，主测试退出码为0。

## 尚未完成

1. 尚无由外部authority生成的生产签名envelope，因此当前只有unsigned build与本地错误签名负例，不声称正式bundle已发布。
2. D21 runner仍需从独立component-dir接口迁移到该joint bundle formal loader。
3. ADV3B02正式重训lineage需先解决`rho_label`比例和checkpoint selection协议冲突。
