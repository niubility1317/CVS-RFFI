# D105-FTU1精确Git archive独立smoke交接

状态：`PASS / LOCAL_TECHNICAL_ARCHIVE_SMOKE_ONLY / NOT_N607_AUTHORIZATION / NO_PERFORMANCE_RESULT`

本次只验证精确Git提交`a0bdbba6bfb56c45682e0c2bde95aa622a68f101`的本地归档与source-only技术闭环。未连接N607，未访问Target、query真值或评分链路，也未创建formal Phase1 asset。

|项目|结果|
|---|---|
|archive|SHA256=`99fd633c78070b940064ca6e95ca9072427457058cab96c3a61e584c7991c0b4`；242913280B|
|成员安全|4763成员、4196文件、567目录；绝对路径、`..`、链接、特殊成员和重复路径均为0|
|四面SHA|54/54项`Git blob=archive member=解包文件=runtime manifest`|
|LF与编译|54/54项LF；独立pyc54/54；源码缓存污染0|
|runtime/method|`873879aad707fd2407b7645de45daa68fec1d3537feaf9fd57fe98b3ab059214`/`7d33662750b160fce82217dace9e1933aa8e43ea2a0df19f59e28adcf8bb4848`|
|checkpoint|SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；195 tensors|
|strict forward|`z_id/pre_relu/z_dom=[2,160]`、float32、finite、ReLU绑定、`eager_forward_hook`|
|legacy隔离|fresh进程前后旧GRB模块均未导入|
|一行source-only export|技术导出与receipt闭环通过；被既有最小34行formal聚合门正确拒绝|
|CLI/测试|9/9帮助面、8/8 FTU1定向、223/223 D105/LPO-RC回归通过|

外部总验证JSON SHA256=`78543dbb00d2ba3381d6e10b9808ebe751e8355d351261fa8c284cbe44c2ba30`；外部完整中文handoff SHA256=`95b4df18c212b959c49942e01c0fcd8a2484fa2387372bc69e4d47b12f3a9441`。本交接不构成N607同步、启动、Phase1封存、Target25或性能声明授权。
