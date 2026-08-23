# FastTrust-RC4-QB E200实施计划

## 追踪清单

|设计要求|代码位置|验证方式|状态|
|---|---|---|---|
|V_cal派生P安全阈值|`code/cvsrffi/muse_ssdg.py`|RC4校准与路由聚焦测试|`LOCAL_VERIFIED`|
|H优先、P填剩余总预算0.15|`code/cvsrffi/muse_ssdg.py`、`code/SSDG/train_ssdg.py`|手工构造路由预算测试|`LOCAL_VERIFIED`|
|N关闭但兼容旧路由|新矩阵和worker参数|矩阵、launcher dry-run|`LOCAL_VERIFIED`|
|U domain/adversarial显式0.16|`code/SSDG/train_ssdg.py`、worker|参数默认和loss分解测试|`LOCAL_VERIFIED`|
|日志张量不保留计算图|`code/SSDG/train_ssdg.py`|requires_grad/grad_fn测试|`LOCAL_VERIFIED`|
|非有限批次技术保护|`code/SSDG/train_ssdg.py`|阈值边界测试|`LOCAL_VERIFIED`|
|重型评估200→56次|新矩阵、worker|调度计数测试|`LOCAL_VERIFIED`|
|单GPU单进程|新launcher|shell dry-run和静态断言|`LOCAL_VERIFIED`|
|最小三行E200矩阵|新config/launcher|矩阵解析测试|`LOCAL_VERIFIED`|
|Phase1权限和Core90调度不变|训练逻辑、worker|协议回归和真实无query smoke|`LOCAL_VERIFIED`|

## 实施顺序

1. 在`test_fasttrust_rc4.py`和`test_phase1_fasttrust_speed.py`加入质量阈值、总预算、日志detach、非有限保护、56次重评估测试并观察RED。
2. 修改RC4校准包和路由，保持旧参数的兼容路径。
3. 修改训练参数、遥测写入和非有限保护。
4. 扩展worker的显式环境参数，新增三行配置和不可覆盖launcher。
5. 完成GREEN、相邻回归、真实checkpoint无query smoke和一次P0/P1审查。
6. 更新追踪状态与最小run报告，提交、push、核对远端OID。
7. 完成N607最小发布流程并启动三行E200矩阵。

## 本地验证记录

- RED：聚焦集合先因缺少质量预算函数、稳定性参数和新矩阵/launcher失败，确认测试能捕获目标缺口。
- GREEN：FastTrust-RC4、加速调度和QB launcher共28项聚焦测试通过。
- 相邻回归：RC4、MUSE集成、协议、satellite、launcher与速度相关11个测试文件共143项通过。
- 真实Core90无query smoke：严格重建`missing=0/unexpected=0`；输入仅为42条source形状样本，`query_input_count=0`、`target_truth_read_count=0`、`target_eval_count=0`；63组有限梯度，遥测张量均已detach；技术覆盖分支满足有效权重`2.067908<=6.3`。
- 生产校准在该小型合成source夹具上对P路由安全失败关闭（P=0、R=42）；这不是性能结果，也不修改正式矩阵阈值。
- 独立P0/P1审查首轮发现1个P1：QB1未与QB2共享0.15总有效身份预算，会混入H预算变化。已先观察定点RED，再把QB1修正为0.15；定点复审确认`[QB0,QB1,QB2]=[0.0,0.15,0.15]`且参数原样传递，最终`P0=0、P1=0`。
