# SF-TAPFT V2并行容量矩阵发布报告

## 结论

第一波矩阵的15个独立support-only候选已全部启动，配合GPU0上的R0占满8张RTX 3090的16个训练槽。候选复用同一CORE90正式bundle、receiver`20-1`、旧类6类、K=10、60条support、capsule、split和seed；query与truth保持关闭。当前最高状态为`RUNNING`，尚无正式性能结论。

## 实现与验证

- Git提交：`f4cb02970b651bd841cf4de3d7f4e7830e566a1f`。
- 新增P0–P4嵌套可训练层配置：head-only、head+norm、time Adapter、完整`t3`、时间/融合高容量层。
- P4只增加`t2.pw`、`time_fuse.0`、`fuse`和`cls_head.id_proj`，未开放frequency或domain分支。
- TDD先验证新测试因接口缺失而失败；实现后58项聚焦测试通过。
- 15份配置均通过严格loader解析；一次发布归档、一次远端编译和真实checkpoint短smoke在N607发布阶段执行。
- release归档SHA-256为`2733163fc745ff45c926e8012bf353c2b44bffd66bb50767121926c4aa4d29f1`，本地与N607一致；远端编译通过。
- P4真实checkpoint 15步无query smoke通过：60条support、4折OOF、全support refit和V2 bundle闭合，source/query/truth均未打开。
- M01–M15的15个PID均存活，CWD全部绑定同一release checkout；每个cmdline中的config和output与对应run ID一致。
- GPU回读为每张卡2个训练进程，无第三个进程；启动期未发现Traceback、Error、Exception或OOM指纹。

## GPU分配

|GPU|槽1|槽2|
|---|---|---|
|0|现有R0|M01 P0_HEAD_ONLY|
|1|M02 P1_HEAD_NORM|M03 P2_R32|
|2|M04 P3_R32|M05 P4_R32|
|3|M06 P3_R16_KD010|M07 P3_R16_RHO075|
|4|M08 P3_R16_RHO100|M09 P2_R16|
|5|M10 P3_R16_C300|M11 P3_R16_C500|
|6|M12 P3_R8|M13 P3_R4|
|7|M14 P4_R16|M15 P4_R8|

每个run的完整预登记字段位于`E:\type10-7\automation_reports\CV-SincNet\<run-id>\report.md`。
