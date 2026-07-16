# Stage2实验优先协议优化追踪

日期：2026-07-16

目标：优化`项目.md`中的Stage2说明。继续严格禁止Phase2接触clean样本、源域样本及其衍生feature/logit/prototype/cache/adapter等状态，同时把协议检查收敛为一次性准入门禁，防止已满足边界的任务反复审计前期工作而不进入实验优化。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| S2-EF-01 | 用户要求：禁止clean访问 | Phase2运行时不得接触clean样本及任何clean-derived signal | `项目.md` | verified | 第7.1节保留硬禁令；第7.4节禁止用clean/source payload或衍生物补准入证据 | 保留sealed Phase1 checkpoint唯一例外 |
| S2-EF-02 | 用户要求：禁止源域访问 | Phase2运行时不得接触source样本、source feature、统计、cache、prototype、adapter等额外衍生物 | `项目.md` | verified | 第7.3节字段与第12节自动化检查覆盖sample/cache/label/derived/replay/adapter | checkpoint之外全部禁止 |
| S2-EF-03 | 用户要求：避免反复纠结前期工作 | 使用现有证据回答三个最小二元问题；输入边界未变时复用，不重复审计Phase1/source/clean历史 | `项目.md` | verified | 第7.4节最小准入、PASS复用和实质边界变化才重审规则 | 协议合规是可行性约束，不是优化目标 |
| S2-EF-04 | 用户要求：实验与优化落地 | 准入PASS后必须进入候选、矩阵、验证和Runner gates；协议说明或报告润色不能替代实验推进 | `项目.md` | verified | 第7.4节实验推进完成定义；第10.3.1与第12节接入 | 远端执行仍服从安全、容量与任务授权 |
| S2-EF-05 | 项目版本规则 | 根目录非Git文件与Git承载面保持一致，仅提交本次项目协议与追踪记录 | 根目录`项目.md`、Git镜像`项目.md`、本文件 | verified | 两份`项目.md`SHA256均为`8CF746DAB8A0E7333AC9BB01925BC53D5E20D6BF3166D5043BACBCD849A170C1`；`git diff --check`通过；已定向提交 | 未覆盖工作树中其他未归属修改 |
| S2-EF-06 | 用户澄清：简化流程 | 不新增独立admission JSON、组合哈希或治理工程；把前期检查压缩为最小三项二元门禁，通过后立即实验 | `项目.md` | verified | 第7.4节明确不新增准入系统，且自动化规则改为复用现有证据 | 不放宽第7.1至7.3节安全边界 |

## 遗漏风险

- 只写“禁止source”但未覆盖source-derived feature、统计、cache和额外adapter。
- 把sealed Phase1 checkpoint误判为可继续查询的source状态。
- 准入PASS后仍以补manifest、写审计报告或追溯旧artifact为主要产出。
- 单个row准入失败时阻塞整个lane，掩盖其他已满足条件的launchable row。
- 只修改根目录非Git文件，未同步Git承载面。
