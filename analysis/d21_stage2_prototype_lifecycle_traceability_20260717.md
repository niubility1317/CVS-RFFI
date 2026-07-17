# D21 Stage2原型生命周期追溯表

## 实现边界

本模块只实现LEO_weak support侧的旧类snapshot与新类append-only注册，不打开query或scorer。当前production runner仍等待ADV3B02 runtime+v2组件外层联合seal；因此本表的`verified`表示本地机制与负向测试通过，不表示正式Stage2性能已验证。

|ID|需求|实现|证据|状态|声明边界|
|---|---|---|---|---|---|
|D21-LC-01|Stage2-B旧类状态只生成一次|`fit_old_snapshot`生成只读prototype、radius与激活掩码|旧状态不可变与bitwise score测试|verified|仅D21内部target-score路径|
|D21-LC-02|旧support不可被同label/K的其它feature替换|state绑定规范化support内容SHA、authority receipt和before capsule root；注册时从实际support重算|内容篡改与receipt错配负向测试|verified|正式root仍由外层authority提供|
|D21-LC-03|Stage2-C只能追加新类|旧prototype、radius、mask与score snapshot位级锁定；after capsule root/receipt写入current state|append-only、重复注册、旧score锁测试|verified|未声称DALI已接入|
|D21-LC-04|基础新类原型不得直接侵入旧类|每个new及组合append前相对Stage2-B纯余弦旧snapshot检查逐类accuracy与最差margin|copy-old对抗注册被拒绝|verified|support非退化不等于query性能保证|
|D21-LC-05|半径不能产生类尺度偏置|K≥5先做纯余弦→radius逐类及组合守门，逐类`radius_active`；旧mask锁定|radius负向、mask与fallback测试|verified|K1统一关闭radius|
|D21-LC-06|K1不得以同一样本拟合并自证|旧类、新类radius与boundary全部关闭|K1全off测试|verified|纯余弦单中心|
|D21-LC-07|新旧/新新碰撞边界稀疏且support-only|每个新类至多1个rival；报告old→new、new→old、new→new并做组合守门|稀疏性、顺序不变与fallback测试|verified|K1不构造boundary|
|D21-LC-08|无query fit、角色Oracle、类别配额或全局分配|公开score API逐样本面对全部注册类；runner无query/scorer入口|API签名和CLI负向测试|verified|正式prediction/scorer隔离待后续矩阵|
|D21-LC-09|support-only状态不能由公开参数自报正式|公开评估器固定输出synthetic状态；sealed状态仅能在runner内部完成authority与materialization核验后生成|公开调用面与mocked materialization测试|verified|真实外层联合seal尚未完成|
|D21-LC-10|部署资源按真实交付物计|分开记录numeric state、metadata、support audit与实际落盘artifact；fixed-point计算含COMMIT总字节|文件stat与COMMIT一致性测试|verified|尚未测N607延迟/显存|
|D21-LC-11|DALI仅作正交候选|runner声明`internal-target-score-lock`、`dali_integrated=false`、`dali_lock_claimed=false`|状态字段断言|verified|待后续接入domain20 max-old scorer|

## 本地验证

- `conda activate ssr-gpu; python -m pytest -q tests/test_stage2_prototype_lifecycle.py tests/test_run_d21_support_only_lifecycle.py`：23项PASS。
- 与Phase1 codec、streaming和exporter联合回归：51项PASS。
- `py_compile`与`git diff --check`：PASS。
- pytest退出后的Windows临时junction `PermissionError`为已知清理噪声，测试退出码为0。

## 尚未完成

1. ADV3B02历史split不满足当前`rho_label=|L_s|/(|L_s|+|U_s|)<=0.1`，正式Phase1 lineage需协议澄清后从头重训。
2. ADV3B02 runtime+v2组件外层联合seal、真实签名与same-fd formal loader尚未实现。
3. DALI尚未接入D21正式score路径。
4. N607真实support-only、开发query和5 receiver×5 confirm seed×3 LEO×new5/10/20矩阵尚未运行。
