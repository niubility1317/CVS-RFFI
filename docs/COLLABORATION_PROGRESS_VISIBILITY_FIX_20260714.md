# Codex过程更新可见性修复交接

日期：2026-07-14

## 问题与根因

近期长任务只显示“运行了多个命令”“编辑了文件”等聚合事件，没有持续显示阶段性工作过程。直接原因是项目根规则、协议源文件及Git发布镜像中同时存在`DO NOT send optional commentary`，该指令会压制原本可用于展示操作、证据和下一步的commentary更新。

截图中的“正在重新连接2/5”表明当时还发生了短时流连接中断，但本机`logs_2.sqlite`只读检查为`quick_check=ok`，没有数据库损坏。该日志库受既有`codex_drop_logs_before_insert`保护触发器约束，当前不保留日志行，因此不能从该库进一步归因具体网络重连。

## 修复内容

- 将根目录`AGENTS.md`和`项目.md`中的抑制规则替换为强制过程更新规则：使用工具或长时间运行时，在首次工具调用前、关键阶段切换、重连或上下文压缩恢复、出现阻塞时，以及持续工作期间至少每60秒发送一次简洁、基于证据的更新。
- 明确过程更新只报告可观察操作、发现和下一步，不披露私有思维链，不倾倒原始日志。
- 同步更新Git发布承载面的`AGENTS.md`、README、精简协议和完整源控制镜像。
- 在活动`CODEX_HOME`的`config.toml`中显式设置`model_reasoning_summary="detailed"`。

## 验证结果

- 活动配置路径：`E:\codex\home\config.toml`，`CODEX_HOME=E:\codex\home`。
- `codex --version`在新配置下正常返回`codex-cli 0.123.0`。
- 全范围检索未再发现`DO NOT send optional commentary`。
- Git承载面目标文件通过`git diff --check`。
- `E:\codex\home\logs_2.sqlite`只读检查结果为`quick_check=ok`，日志行数为0，保护触发器为`codex_drop_logs_before_insert`。

## 生效边界

Codex在任务启动时加载`AGENTS.md`和启动配置。完成本次任务后应完全退出并重新打开Codex，再新建一个使用工具的任务验证；当前已启动任务不会可靠地重新加载整条指令链。产品不会展示逐token私有思维链，修复目标是恢复详细reasoning summary和可核验的中间工作更新。
