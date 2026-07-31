# D105-FTU2实现与本地验证

## 实现结果

`D105-FTU2`只修复Phase1 strict tap的固定批容量执行合同：

- `forward_batch_capacity=256`；
- 每批实际前向shape固定为`[256,2,T]`；
- 末批不足256行时零填充，完成同次D105双分支前向后只保留真实行；
- strict tap receipt升级为v2，绑定capacity、调用数、末批真实行、填充行和固定策略；
- `tap-cache`仅接受原生整数256，并在任何cache/checkpoint路径访问前拒绝其他类型或数值；
- `tap-runtime`保留弃用兼容参数，但默认256且帮助文本明确实际capacity固定为256。

未修改D105 feature tap、模型、checkpoint、reference dual archive或全8400行`max_abs<=1e-5`门。

## 验证结果

|验证面|结果|
|---|---|
|257行边界|2次实际256行前向；末批1行真实＋255行零填充|
|8400行边界|33次实际256行前向；末批208行真实＋48行零填充|
|批形状敏感fake model|固定padding后保留行一致；删除padding会被测试捕获|
|非法tap-cache capacity|`128`、`256.0`、`np.int64(256)`、`True`均在外部路径访问前拒绝|
|receipt v2负测|capacity、调用数、末批行数、策略、bool/零row_count、旧v1均拒绝|
|真实checkpoint CPU smoke|195 tensors；1/208/256真实行相对独立256零填充reference三路最大差≤`1.91e-6`|
|统一回归|10个D105/LPO-RC文件238/238通过|
|canonical identity|54/54 runtime文件通过；runtime=`8797de12f035db609aeb6f453f096571f216d0d514d6705344e763f5ec63a498`；method=`9a87e51de4d775ff2ea05e59654afaa62844edaf2def942d8f73c8e289ea61e6`|

唯一警告仍为`model.py`旧`torch.cuda.amp.autocast`FutureWarning，不影响本次执行合同。

## 文件身份

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d105_phase1_bundle.py`|`21790713b7b5577dd8b7f2612969522d45d6307e5f8a883bafd25319796414c7`|
|`code/scripts/build_d105_phase1_bundle.py`|`1b63f18ecb8ab1daf4de6637aa67db1a5c722fab9e466e095fa1bf7c9ab143ef`|
|`tests/test_stage2_d105_phase1_bundle.py`|`b4f7128e392714bfe0a843708134fe2ecfe17bc49a1bb65110842e5bbd55b5b3`|

本地验证不构成N607同步、Phase1 formal asset、Target25或性能授权。下一门是Git提交后的精确archive smoke和全量source-only GPU parity技术复现。
