# D16-FCAR独立红队复核

## 结论

D16模块协议安全为GO；真实路线性能与formal promotion为NO-GO。最终审查对象为：

```text
module_sha256=A7762ADF749B4FB797F83537EBA525D52E6172C92208BC5B81529F074300FDE1
tests_sha256=72EA09967640D8EC5C7947063ED1CA3DAC73CFA354D713CBE2CCB0B377D94E64
trace_sha256=2E42C31368EC5E04DB17672D71066F9341FE3B1BC00644AB0A5EC31194317055
```

聚焦测试14/14通过，module与runner联合测试19/19通过，`py_compile`和`git diff --check`通过。

## 已关闭的P0

1. 2-fold OOF阈值记录及其peer模型均排除当前held物理样本。
2. 五个outer L2O fold逐一把held2 feature放大`-1000`后，selection SHA、decision tensor SHA、floor handles、enabled mask和train physical-ID SHA保持不变。
3. K1只能构造canonical rank0 true Z0；K2–K4 fail closed；K10 package不能冒充K5。
4. 恶意重算state自封存SHA不能绕过语义门。伪造force-zero正修正、positive-rank K1、K2、candidate/operator不一致、enabled count、amplitude grid和head-op资源字段均被拒绝。
5. floor由support OOF baseline底四分位自动生成，未硬编码当前TX或class handle。
6. predictor入口只接受单个样本并对全部registered classes决策，没有role、quota、batch-count或global assignment接口。

## 保留边界

旧logit逐位锁不等于旧类决策锁。新增类logit仍可越过完全相同的旧类列，因此真实runner必须检查Before-old与After-old的逐类遗忘。D16 outer L2O已输出该证据，runner也实施全部scenario×全部fold硬门；module单独fit仍不可被其他调用方当作promotion许可。

`deployment_state_consistency_veto`使用full-support prototype和full-support envelope，包含被评support自身。它只用于撤销部署状态不一致的修正，不能作为性能证书。唯一无held泄漏的性能入口是严格K10 outer L2O。

单物理样本单LEO overlay、clean/source不可达、enrollment-only角色与跨场景互斥仍由上游sealed package和pre-open validator证明。D16 runner复用D14验证路径，并在真实运行中核对三场景physical sample ID与parent received-IQ SHA零交集；D8b authority仍为`development_diagnostic_only`，所以不得开放query或125矩阵。

## 真实性能判定

真实D8b strict-K10运行最终回退true Z0。`margin_band=0.02`没有改变任何三场景聚合预测；`margin_band=0.04`在low和rain分别改善一个旧类10pp和20pp，但seen-new类下降10–20pp。该结果证明逐类非对称幅度仍缺少old-rival条件化证据，不能同时保护旧类遗忘与new floor。
