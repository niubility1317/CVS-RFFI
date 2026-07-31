# D105 Phase1 R6技术失败交接

状态：\`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT\`

本交接关闭唯一运行\`d105_phase1_sourceheld_a0bdbba6_20260731_r6\`。R6只执行了Phase1 source-held stage1的唯一detach；未访问Target、未启动Target25、未生成预测、truth-open、score、gate、component或formal asset，因此不存在可报告的性能结果。

## 冻结身份

|字段|值|
|---|---|
|run ID|\`d105_phase1_sourceheld_a0bdbba6_20260731_r6\`|
|N607 run root|\`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_a0bdbba6_20260731_r6\`|
|source commit|\`a0bdbba6bfb56c45682e0c2bde95aa622a68f101\`|
|source archive|\`99fd633c78070b940064ca6e95ca9072427457058cab96c3a61e584c7991c0b4\`；242913280B；4763成员|
|runtime/method|\`873879aad707fd2407b7645de45daa68fec1d3537feaf9fd57fe98b3ab059214\`/\`7d33662750b160fce82217dace9e1933aa8e43ea2a0df19f59e28adcf8bb4848\`|
|launcher|\`7f23f6e9bc8038a859962fed4b8fbb6ab63a805301ac9dbae7310a51be36e28d\`；5624B；LF-only；\`bash -n\`通过|
|D102公开撤销输入|\`99393aa21b30cc654ba784ecf8a60b1ac8497e67d7fdefc3ee12872133293734\`/\`53d138b36f7688431d364f2e6291e86aefb9c9fc3e84697e288f0aa813b81c58\`|

## 预启动闭合

|门|结果|
|---|---|
|direct N607 preflight|通过；普通账户、项目根和8张GPU可见；GPU0启动前0%/1MiB且无compute process|
|run root与输入|新run root启动前不存在；4个冻结input远端SHA、大小全部一致|
|archive|单一\`source/\`根；无绝对/逃逸路径、链接、特殊成员或重复成员；解包4196文件/567目录|
|runtime|canonical loader通过；54/54核心文件SHA一致；source外独立pyc54/54通过|
|外部依赖|cache、salt、checkpoint、reference dual archive的SHA全部一致|
|真实checkpoint R2终检|通过；195 tensors；torch=\`2.1.0+cu121\`；loader=\`legacy_pickle_exact_frozen_sha_only\`；\`z_id/pre_relu/z_dom=[2,160]\`、float32、finite、ReLU绑定、\`eager_forward_hook\`；GRB未导入|
|source-only预检边界|1行技术导出只用于入口验证；\`target_rows=query_rows=0\`；正式聚合预期拒绝，最小行数为34|
|CLI与launcher|9/9帮助面、18个输出文件、LF与\`bash -n\`均通过|

归档目录内旧\`verify_loader_and_real_checkpoint.py\`只是已被R2替代的本地archive-smoke辅助脚本，不在冻结\`source/\`、不在R6四个input，也不在launcher/正式CLI入口。它把1行技术导出误送入正式聚合读取而得到\`strict tap feature rows drift\`；该结果不构成R6方法或运行时故障。实际规则来自同一archive：\`DOMAIN_DIM=32\`且\`StrictTapRows\`要求\`rows>=DOMAIN_DIM+2=34\`。R2终检把这一行拒绝作为预期的防误晋级边界，并在同batch2行前向上验证三路特征。

## 唯一启动与失败

R6唯一detach主PID=\`2817802\`，子PID=\`2817808\`。主PID的CWD、cmdline和\`CUDA_VISIBLE_DEVICES=0\`均绑定R6 run root与冻结launcher。启动后首个\`tap-cache\`写出strict tap，再于reference dual byte-parity guard退出；\`pipeline_stage1.exit=2\`。

归一化异常指纹：

\`\`\`text
build_d105_phase1_bundle.py: error: strict D105 tap/reference dual archive parity failed
\`\`\`

只读差分确认metadata没有漂移，8400行\`labels/receiver_ids/physical_ids\`均与reference archive一致；失败来自特征数值超过固定\`1e-5\`阈值：

|比较项|最大绝对差|阈值|结论|
|---|---:|---:|---|
|\`z_id\`vs\`ReLU(pre_relu)\`|\`1.9073486328125e-05\`|\`1e-5\`|失败|
|\`z_dom\`|\`0.0019412636756896973\`|\`1e-5\`|失败|

reference dual archive SHA=\`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0\`；R6 strict tap SHA=\`68c08c85b2fdd7429444f9c9e92859f5e412076165d2e36d4fc634702ae1f5c6\`。这是启动入口的确定性parity故障，不是数据协议、Target结果或模型性能结果。

## 工件、清理与声明边界

|项|计数/状态|
|---|---|
|strict tap|\`5\`个文件；8400行；source-only；hook、pre-ReLU与z_dom收据存在|
|prediction/truth-open/score/gate|\`0/0/0/0\`|
|component/formal asset|\`0/0\`|
|Target/Target25/authority seal|均未访问或启动|
|主/子进程|均已退出；未执行kill|
|GPU0|终态0%/1MiB；无compute process|
|本地SSH|每次短连接后\`ssh.exe\`和N607:22 established连接均为0|

回收目录为本目录的\`artifacts/\`。关键回收哈希：主日志=\`e11c6054150e9cf008890dfd61cca707b2b3afc16d356b8b288f01e01f0c1789\`，exit=\`53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3\`，R2终检=\`165c28af1eacb2935e2466439e9c07a90962058bd51829818c60cf07e49192af\`，R6 strict tap receipt=\`7b2d33fb970dd119851444a2997d21b1235405b1a24ce62fff0964f98b268059\`。本地回收文件均与远端SHA匹配。

后续只能在本地诊断reference dual archive与当前冻结生产tap的数值parity来源，完成独立审查、Git提交和全新不可覆盖run ID后，才可重新进入Phase1 release；不得恢复、覆盖或重试R6。
