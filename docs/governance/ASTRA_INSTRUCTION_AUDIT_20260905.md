# GPT-6 Astra指令与工作流审计

日期：2026-09-05。范围：E:/type10-7当前控制面与CVS-RFFI Git发布仓库。
结论：将常驻AGENTS、技能入口和实验执行说明分层，移除重复流程与旧optimizer强制门槛，保留科学权限和操作安全。本次是指令治理优化，未进行模型速度/质量A/B测试，不声明性能提升比例。

## 官方依据
已通过OpenAI官方文档服务检索并阅读全文：
- [GPT-6 Astra模型指导](https://developers.openai.com/api/docs/guides/latest-model)：明确行动授权、技能冲突优先级、适度验证、按任务安排协作。
- [Codex AGENTS.md](https://developers.openai.com/codex/guides/agents-md)：按目录加载，默认项目指令预算32KiB；精简常驻入口，重启/新运行重建指令链。
- [Codex Skills](https://developers.openai.com/codex/skills)：以name/description发现，命中后加载SKILL.md；支持仓库.agents/skills，引用按需读取。
这些是本次设计依据；具体八项流程、数据边界和N607限制来自用户项目约定，不是官方模型限制。

## 审计范围与发现
已检查当前AGENTS、项目协议、optimizer控制manifest/contract、相关preflight实现及测试、仓库技能目录与相关全局技能。Git跟踪清单中无原有SKILL.md或GitHub Actions workflow；当前根.agents未发现技能文件。旧快照、日志、数据及临时pytest缓存不作为活动控制面；初次文件枚举遇到两个缓存访问拒绝，不将此声称为全盘扫描。
发布主检出work/cvs-active存在大量其他任务暂存与未暂存改动，因此从d994f419创建独立分支codex/astra-instructions-20260905；未操作其index或切换其分支。

| 问题 | 处理 |
|---|---|
| AGENTS重复规定Git、SSH、数据复验、审查、结果交付 | 收拢一次定义，根入口按任务链接 |
| 禁止pwsh与默认pwsh并存 | 保留用户的禁用要求，Windows原生优先 |
| 全局N607技能强制所有Windows操作经Git Bash | 仓库明确覆盖此旧建议，只在.sh/POSIX/MSYS2任务使用 |
| optimizer强制64行、额外seal/hash/authority及多角色审查 | 新流程直接落实八项白名单，小矩阵先行 |
| 历史prompt/state被当成当前授权 | manifest明确历史证据与当前目标、运行证据的边界 |
| 检查状态可能扩大成自动修复 | 明确周期监控/修复授权与一次性只读状态检查的区别 |
| 文本测试要求多份控制文件重复旧指令 | 删除7项过时文案一致性测试，保留55项工具行为回归 |
| Git协议镜像落后于本机项目.md | 原文同步当前科学协议到docs/PROJECT_PROTOCOL.md；未修改科学语义 |

全局插件缓存和全局技能文件不在当前仓库修改范围，未批量改写。superpowers的广泛强制触发、固定测试/审查流程存在过度约束风险；本项目以用户范围、按需技能和八项白名单处理冲突。记录REJECTED_EXTRA_GATE：不将技能写作中的多轮压力测试/固定重复审查转成此次项目治理交付的前置条件；实施一次有界独立审查和聚焦验证。

## 交付与职责
- AGENTS.md：常驻任务约定、按需路由、Windows/Git和关键权限。
- .agents/skills/cvs-experiment-workflow/SKILL.md：新建仓库级技能，按设计/监控/修复/评分选择入口，不复制全套规则。
- tools/optimizer_workflow_contract.md：八项生命周期、完成条件与四态指标。
- tools/optimizer_control_manifest.md：唯一职责映射，旧optimizer工具改为显式历史兼容用途。
- docs/workflows/n607.md：连接、进程保护、技术修复与重发。
- code/tests/test_optimizer_workflow_tools.py：只删除7项重复旧文案检查。
- docs/PROJECT_PROTOCOL.md：当前协议镜像。
- docs/governance/astra-20260905-before/：保留非Git根目录原3份控制文档供回滚/审计，非活动指令，平时不加载。

## 大小与预期作用
按UTF-8字节计算，不把字节等同token或模型能力：
| 文件 | 根目录原始字节 | 精简字节（验证时） |
|---|---:|---:|
| AGENTS.md | 31720 | 6054 |
| optimizer_control_manifest.md | 15344 | 2184 |
| optimizer_workflow_contract.md | 56081 | 4911 |

AGENTS减少约81%，三份活动文档合计减少约87%。新增按需技能约1.7KB、N607操作约4.7KB；归档不自动加载。预期减少上下文占用和规则冲突，不保证任务耗时或准确率提升。
保留当前宿主的模型及reasoning设置；未把API参数建议强套给桌面应用，未创建新的模型调用或消耗额度的benchmark。

## 验证与边界
- ssr-gpu下55项optimizer工具回归通过；解释器与CONDA_PREFIX均确认属于ssr-gpu。
- 官方skill-creator的quick_validate通过。
- 新活动入口文件UTF-8、无替换字符、引用路径检查通过。
- 独立审查发现一次性监控权限歧义，已按原规则收紧；其他主要权限/完成条件未发现遗漏。
- JSON只读解析现有optimizer state成功；不采用其旧规则作为新权限。
- 旧标识cv-sincnet-n607-monitor-optimizer-v4对应本机automation.toml不存在；未声称实际调度器迁移成功。其他已保存自动化不在本次仓库改动内。
- 旧optimizer Python实现及历史prompt/state保留，未把其旧schema重写为新实验平台；新任务走当前候选launcher与本次最小流程。
- 本次无N607访问、启动、停止或远端文件修改；没有运行科学实验。
- 发布使用独立分支与PR，主检出无关脏文件保持原样。最终commit、远端OID及PR以本次交付回执为准。

## 生效与复核
当前工作区根控制文件同步新版本；Git承载在本次分支。发布主检出的集成需正常PR合并，不能在存在他人暂存内容时强制切换或重置。
新Codex运行从对应目录加载更新后的AGENTS；已有会话中注入的旧指令可能继续存在，应新开任务验证实际加载。技能结构有效不等于已经证明宿主自动发现或运行性能改善。
后续可用同类真实任务记录完成率、澄清次数、重复验证次数、时间和token，比较旧/新指令；不把评测本身加入日常实验gate。
