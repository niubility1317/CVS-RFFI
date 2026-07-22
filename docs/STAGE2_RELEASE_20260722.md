# CVS Stage2研发快照发布记录

日期：2026-07-22

## 发布范围

- Stage2隔离执行与Landlock预运行证据实现。
- D4A–D10单观测、support-only、旧类锁定、局部边界、稀疏operator融合与盲接收侧operator bank实验性模块。
- qKNNV42 125计划的Stage2-B reference-query和远端路径约束修正。
- 对应单元测试、设计追踪、实验进度与任务记录。
- 本分支在发布前已积累的D11–D91代码、实验报告和诊断结论。

## 证据边界

- `p2_min_v1`、`VALIDATED_ONCE`、单物理样本单LEO接收观测、support/query物理ID互斥和逐query全注册类决策边界保持不变。
- D4A–D10追踪文件中的`blocked`、`rejected`和diagnostic-only状态原样保留，不因上传GitHub而转为正式性能证据。
- D92 Role-Oracle 125上限实验保留在独立分支，标记为`LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE`，不得并入协议合法晋级结论。
- 不上传数据集、checkpoint、模型权重、原始大日志、`runs/`、海量`local_artifacts/`或N607凭据。

## 本地验证

- Conda环境：`ssr-gpu`。
- `python -m py_compile`：47个当前变更Python文件通过。
- 聚焦pytest：21个测试文件，共131项通过。
- `git diff --check`：通过。
- 发布前执行敏感标记扫描；未发现令牌、私钥或密码内容。

## GitHub承载

- 仓库：`niubility1317/CVS-RFFI`。
- 主发布分支：`codex/cvs-rffi-release-20260626`。
- 草稿PR：`#2`，目标分支`main`。
- D92独立分支：`codex/d92-role-oracle-125`。
