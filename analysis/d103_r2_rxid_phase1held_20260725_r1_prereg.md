# D103-R2正式Phase1-held预注册交接

状态：`PREREGISTERED / INPUT_HASH_RESOLUTION_ONLY / N607_LAUNCH_NO_GO / TARGET25_NO_GO`

实验ID：`d103_r2_rxid_phase1held_20260725_r1`

完整本地报告：`E:\type10-7\automation_reports\CV-SincNet\d103_r2_rxid_phase1held_20260725_r1\report.md`

## Release证据

- candidate：`D103-R2-RXID-CROSSRECEIVER-MB4`
- 实现commit：`59978e44`
- 独立复审index SHA256：`30c8c98ff8fcdf2915f4c2e797c605cecc98d138e95ad3b0bc6e542faf9fdc9b`
- release复审：`REVIEW_GO / P0=0 / P1=0 / P2=2`
- 本地验证：61项D103定向测试通过；36个Python文件编译解析通过；真实tap/dual 400step无query-truth smoke通过。
- 无Git push或远程Git上传。

## 冻结实验

- `protocol_schema=p2_min_v1`
- source split：8400→588/5292/2520；42个receiver×TX组各`L_s=14`；leave-day后10–12。
- fit：1 final+49 outer+196 leave-day=246fit、98,400step。
- held：63性能行、49稳定性行，每个稳定性行4个实际160维shift。
- matched M0/D102/D103；D102保持`DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE`。
- 每GPU最多2个worker；总GPU时≤30、显存≤4GiB/fit、run-root≤20GiB。
- 不读取Target；held接受前`TARGET25_NO_GO`。

## 首次N607交接权限

唯一运行代理`scxmap_held_n607_runner`首次只允许执行direct preflight、GPU/进程/磁盘检查、冻结输入路径/SHA核对和远端同名release文件冲突检查。必须只读解析：

`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_train/cache_set.json`

的当前SHA256并返回主代理。该SHA写回完整报告并提交前，禁止sync、创建run-root或启动进程。

## 停止规则

不依据性能停止。仅在P0协议/安全错误，或两个不同fit在prediction前产生同一规范化确定性异常指纹时，停止dispatch并终止三重绑定的本run PID。保留全部partial artifacts，不自动重试，不覆盖原run。
