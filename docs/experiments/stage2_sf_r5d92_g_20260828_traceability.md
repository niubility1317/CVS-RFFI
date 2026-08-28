# SF-TAPFT-R5D92-G设计追踪

设计来源：`E:\codex\home\attachments\9d460bf2-b479-404b-b80d-f95bc4786b59\pasted-text.txt`

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|R5-01|第五节阶段A|K=10旧类support按类确定性划分为5个互补留二折，每轨迹每类8-shot，每样本恰好留出一次|`code/cvsrffi/target_only_progressive_adapt.py`、`tests/test_stage2_sf_r5d92.py`|implemented|聚焦测试通过|不增加K；原型初始化也严格排除本折held-out|
|R5-02|第五节阶段A|每轨迹仅训练target head和`t3.norm(weight,bias)`，保存250/350/520步|`code/cvsrffi/target_only_progressive_adapt.py`|implemented|聚焦测试通过|保持1152个在线可训练元素|
|R5-03|第五节阶段B|按步平均5条许可delta，alpha仅取0.75/1.0，形成6个候选|`code/cvsrffi/target_only_progressive_adapt.py`|implemented|聚焦测试通过|只聚合共享`t3.norm`锚点；辅助head不跨折平均|
|R5-04|第五节阶段B|以MacroCE、class-CVaR和margin regression进行旧类便宜筛选，只保留Top-2|`code/cvsrffi/target_only_progressive_adapt.py`|implemented|聚焦测试通过|风险来自真正held-out support，不读取query|
|R5-05|第六节|对Top-2与R0基线执行D92 E0旧新类support cross-fit，输出old pre/post、new、H、F、min-old、min-new、old→new|`code/cvsrffi/stage2_sf_r5d92.py`、`tests/test_stage2_sf_r5d92.py`|implemented|聚焦测试通过|D92 E0支持K8 cross-fit，未复制support|
|R5-06|第七节|执行可行性硬约束、按LCB(H)排序，并在无候选通过时回退R0→D92|`code/cvsrffi/stage2_sf_r5d92.py`|implemented|聚焦测试通过|协方差失败明确拒绝；不使用随意加权总分|
|R5-07|第八节|最终丢弃SF target head，仅保留适配后identity表征并重建D92 E0统一头|`code/cvsrffi/stage2_sf_r5d92.py`、`code/cvsrffi/stage2_sf_erbt_four_state.py`|implemented|四状态聚焦回归通过|query只使用单一D92仿射头|
|R5-08|第九节|适配后重新估计中心、共享协方差与块尺度，记录identity/FFT trace、条件数和正定性|`code/cvsrffi/stage2_sf_r5d92.py`|implemented|聚焦测试通过|当前实验使用D92 E0无RF32 160+96维路径|
|R5-09|第十至十一节|记录墙钟、轨迹/fit/forward计数、参数/状态/cache、RSS和GPU资源；300秒方法预算|runner、config、实验报告|partial|240秒软边界已测试，真实资源待实验|系统技术失败才停止|
|R5-10|第十三节|同row验证J0 D92、J1 R0→D92、J2 R3→D92、J3 R5D92-G；四状态truth-last评分|`code/scripts/run_sf_r5d92_adapt.py`、四状态runner、实验报告|implemented|入口聚焦测试通过，真实闭合待实验|J0复用每行共同`DA0`；首轮单seed/new5；J4为宽权限历史上界|
|R5-11|第十二节|MRIOR地面教师蒸馏初始化|后续独立地面阶段|deferred|未进入本轮|需要新的Phase1地面教师训练，不属于本轮source-free星上候选最小验证|
|R5-12|第十三节J4|MRIOR-SDA→D92 E0宽权限上界|历史MRIOR正式报告|deferred|复用历史证据|权限和训练输入不同，不混入source-free晋级；不为本轮重复训练|

## 本轮实验锁

- 运行矩阵：J0、J1、J2、J3；receiver=`20-1`、seed=`713101`、K=10、new5、三种互斥`leo_*_weak`场景。
- 数据：复用匹配的`p2_min_v1`、`VALIDATED_ONCE` capsule/split；不因方法变化重验数据。
- query边界：prediction完成前不读取query truth/role；scorer只在四状态prediction闭合后连接truth。
- 技术停止：错误数据句柄、support/query重叠、错误checkpoint/场景/seed/K、新输出覆盖、确定性异常或无prediction闭合；不得因低性能停止。
- 性能判定：以同row`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`和四类因果差值分析；REG0新类指标为`N/A`。
