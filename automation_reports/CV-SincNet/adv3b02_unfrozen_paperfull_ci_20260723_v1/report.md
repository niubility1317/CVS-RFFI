# ADV3B02可训练骨干上的CSIL与MoPC-HR全量论文方法对比

## 状态与边界

- 实验ID：`adv3b02_unfrozen_paperfull_ci_20260723_v1`
- 状态：`LOCAL_IMPLEMENTED_INTEGRATION_VERIFIED_REMOTE_SMOKE_PENDING`
- 类型：`FORMAL_PAPER_METHOD_COMPARISON_BASELINE`
- 对比方法不受Stage2主方法资源和`p2_min_v1`权限上限约束；全部新类注册support与新类评测query必须叠加LEO星地信道。
- ADV3B02替换论文私有编码器，因此是paper-mechanism-full CVS adaptation，不是原网络逐层同构或官方数值等价复现。

## 矩阵与参数

- 矩阵：5 receiver×5 seed×K=`1/5/10/20`×new=`2/5/10/20`×2方法=800cell；三LEO场景共2400行。
- CSIL：3epoch、batch20、`0.01/(1+0.01*iteration)`、momentum0.9、L2=0.05、KD=0.2、EWC=1、zero-bias与旧fingerprint块mask。
- MoPC-HR：20epoch、batch16、SGD lr0.01、momentum0.9、weight decay2e-4、noise std0.05、α=0.97、paper cosine、语义层平方L2 HR、β/λmax=1。
- MoPC new10/new20分别按2×5/4×5顺序增量；纠正原型进入下一stage replay，最终query走当前模型全注册类classifier logits。

## 追踪汇总

|状态|ID|
|---|---|
|verified|TR-01、TR-02、TR-04、TR-05、TR-06、TR-07、TR-10|
|implemented，待N607真实数据/运行证据|TR-03、TR-08、TR-09、TR-11、TR-12|
|pending|TR-13、TR-14|

完整逐项追踪、验证命令和后续N607证据同时维护于工作区主报告：
`E:\type10-7\automation_reports\CV-SincNet\adv3b02_unfrozen_paperfull_ci_20260723_v1\report.md`。
