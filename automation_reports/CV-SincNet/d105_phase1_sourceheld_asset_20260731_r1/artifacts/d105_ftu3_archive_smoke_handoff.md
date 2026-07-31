# D105-FTU3精确archive独立验证交接

状态：`PASS`

审查：`P0=0 / P1=0 / P2=0`
声明边界：`NOT_N607_AUTHORIZATION / NO_PERFORMANCE_RESULT`

## 核心结论

- 精确source commit：`230c6cbc9149250ca0303ca240945d0e0992360e`；证据HEAD：`a4e14e83235dc33c8287500cb0540234d6201ea6`。source是证据HEAD祖先，直接从source commit生成archive，未checkout；工作树前后均clean。
- archive：`E:\type10-7\automation_reports\CV-SincNet\d105_phase1_sourceheld_asset_20260731_r1\archive_verify_230c6cbc\source_230c6cbc.tar`，243005440bytes，SHA256=`16d57519cfa15d9929a38282217b0a2e2908e5c92e8b42672dae1537386855c7`。
- tar安全门通过：单一`source/`根，共4775个成员（4206个普通文件、569个目录）；绝对路径、反斜线、`..`逃逸、错根、重复、软链、硬链及特殊成员均为0。
- canonical runtime=`5de5926bbb2e9fd78b2f3315ec6e109964ddd6216ebe4f75e428b6b9f6bf11bc`，method=`7345f81e88588c46ad453eb315786306f28291478a5eaddce618ef7ee6998ecd`。相对FTU2只变更`stage2_d105_phase1_bundle.py`的core hash及method→runtime绑定；方法参数和阈值未变。
- 54/54个runtime文件满足Git blob=tar成员=解包文件=manifest；LF54/54；隔离pyc54/54；解包源码零污染。
- 9/9个正式help返回0；FTU3定向15项和固定10文件253项回归均执行到100%。
- signed integer修复门通过：7个普通计数字段仍要求原生非负int；两个`*_min_net_correct`允许原生负int；bool、float、numpy.int64拒绝。负证据仍为`DIAGNOSTIC_STATUS`、`all_noninferior=false`，正式seal被拒绝。
- 真实checkpoint固定256 CPU smoke通过：195个state tensor，eval=true，state前后不变，GRB未导入；1行→1次forward、末批1实+255补零、maxabs=0；208行→1次forward、末批208实+48补零、maxabs=0；256行→1次forward、末批256实+0补零、maxabs=0。独立reference使用`bytearray/frombuffer`，未直接调用`torch.from_numpy`。
- 8400仅为形状敏感fake：33次forward，末批208实+48补零，每次forward shape均为`[256,2,1]`；未使用真实8400行IQ。

## R7和后续边界

- R7永久状态保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。其产物绑定旧FTU2 runtime/tap hash，本次未作为新输入、未读取或报告性能字段、未绕过旧hash创建新正式条目。
- 本机未做完整8400真实source IQ reference parity。新N607第一次tap仍必须先通过完整reference parity硬门，之后才能考虑正式Phase1资产。
- 本交接不授权N607连接、上传、启动或任何性能/正式资产声明。
