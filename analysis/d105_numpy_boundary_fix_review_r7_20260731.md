# D105 NumPy/Torch边界修复独立发布审查R7（2026-07-31）

## 审查结论

**GO（仅限本地实现与证据链就绪）**。本审查发现`P0=0`、`P1=0`。该GO不授权N607同步、启动、重启或性能解释；R4的不可变终态仍为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

审查对象为工作树`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`中相对基线`32240ff984306af8e0cdd305621e4f53dd5d1e60`的未提交D105边界修复。审查只覆盖本地diff、R4回收工件和本地验证；未连接N607、未修改实现/配置/测试/主报告、未启动实验、未提交或推送。

## R4根因与证据边界

- 已逐字节复核R4交接`handoff.md`：SHA256=`f362f5051a71d0dd88552a815c2a82680b6157a537ef1ebc36b1d8e720a3811a`。
- 已逐字节复核完整`pipeline_stage1.log`：992B，SHA256=`9aa65a3e034283ee481b0bb12792ac6fe8fd6e51fb3b71470abe259de49ffe10`；调用栈明确落在旧`stage2_d105_phase1_bundle.py:1658`的`torch.from_numpy(batch)`，异常为`TypeError: expected np.ndarray (got numpy.ndarray)`。
- 旧路径在此调用前已检查source IQ的float32、三维、`[N,2,T]`、`N>0`与有限性，并在每批调用`np.ascontiguousarray(...,dtype=np.float32)`。因此本异常不支持“错误dtype、shape、非有限IQ或非连续批次”这一数据根因。
- R4预启动记录为PyTorch`2.1.0+cu121`、54/54运行时文件和195个checkpoint tensor通过；失败后仍54/54匹配，Target访问为false，严格tap、预测、truth-open、评分和gate工件均不存在。故证据足以支持“该部署环境中的NumPy→Torch ndarray桥接不兼容”这一受限结论，不把内部C-API缓存/ABI的具体实现细节当作已直接证明的事实。

## 逐diff审查

|审查面|独立结论|
|---|---|
|输入桥`_tensor_from_d105_float32_c_iq`|只接受精确`np.ndarray`、native`float32`、`[N,2,T]`、`N/T>0`、C连续且有限输入；非法类型、dtype、形状、字节序、布局和非有限值均在桥接前失败。`frombuffer(count=size)→reshape→clone→to`保持C序字节和shape；`clone`切断NumPy寿命/别名。`RuntimeError`、`TypeError`与`ValueError`归一化为调用面的D105错误，不使用try-fallback放宽非法输入。|
|可写/只读与设备|红队实测可写和只读C连续buffer均按字节复制，输入在返回后变化不影响tensor；CPU和本机可用`cuda:0`均得到目标设备tensor。大端float32被严格拒绝，避免原始字节被误解释。|
|输出桥`_to_numpy`|固定`detach().cpu().contiguous()`后用`tolist()→np.asarray(...,float32)`，不再调用`Tensor.numpy()`。IEEE-754 binary32的有限值可被Python binary64精确表示，回转至float32保持值和位模式；实测`+0/-0`、正负最小subnormal、最大subnormal、最小normal及正负最大有限值字节一致。NaN和正负Inf在输出前拒绝，随后复核shape和有限性。|
|资源与持久化|`tolist()`只作用于已CPU连续的单个tap tensor；严格tap批上限为256，最大hidden宽度320，临时Python对象峰值为单batch量级，不持久化IQ，也不改变tap输出/receipt语义。|
|正式执行面与导入闭包|Phase1严格tap和Target25`_tap_rows`均改用同一helper。对D105代码与脚本搜索后，已无可执行`torch.from_numpy`或`Tensor.numpy()`残留。query evaluator到Phase1 bundle是单向受控依赖；Phase1 bundle没有反向import query evaluator，清单中的query字符串只是受控entrypoint登记。两模块均位于54文件hash闭包内。|
|协议与方法锁|diff未增加query truth、role、class quota、跨query状态或Target越界输入。canonical method loader仍精确锁定`K={1,5,10}`、`M0/M_DA/M_HEAD/M_JOINT`、`query_state_updates=0`、Target25的25外层row/三LEO场景及`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`。|

## 独立验证

|验证|结果|
|---|---|
|4项新增最窄pytest桥接回归|通过：Phase1输入字节/去别名/注入`from_numpy`失败、非法输入拒绝、Target25绕过故障接口、输出绕过`Tensor.numpy`失败。|
|独立红队边界脚本|通过：精确数组、float32、`[N,2,T]`、C连续、有限性、N/T下界、read-only buffer、非别名、native/大端字节序、CPU/实际CUDA设备和异常转换；输出的signed zero、subnormal与有限极值字节一致，NaN/Inf拒绝。|
|D105统一关键回归|9个D105测试文件100%通过；唯一输出为既有`code/model.py:693`的`torch.cuda.amp.autocast`弃用警告。|
|54文件编译与闭包|54/54受控文件以内存Python编译通过；canonical runtime/method loader通过并验证全部文件哈希。|
|差异卫生|相对基线的`git diff --check`通过。|

最终canonical身份：

- runtime SHA256：`8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425`
- method SHA256：`f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e`

## 问题清单与后续边界

|等级|结论|
|---|---|
|P0|无。|
|P1|无。|
|P2-1|`code/model.py:693`的`torch.cuda.amp.autocast`弃用警告在统一回归中仍出现，属于本次diff外的既有兼容性提示，不影响本审查结论。|
|P2-2|本地环境不能替代R4的Torch2.1/NumPy2技术健康验证。若后续申请新的N607 release，必须先完成Git提交、不可覆盖新run ID和专属runner预检；该验证只能证明执行健康，不能产生或解释性能结果。|

本回执保持未提交，交由主代理连同经审查的D105修复统一提交。
