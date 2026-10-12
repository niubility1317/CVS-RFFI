# 唯一独立 P0/P1 审查

PASS，P0/P1为0。Reviewer：`/root/wave_geometry_p0p1_review`。

六行scratch30×222=6660更新，旧E20配置仅静态架构/原始数据加载，权重祖先为空。CE无U；SELF前1110更新纯CE，后当前student eval detached软目标，95%与相邻完整pass两次稳定门控、std.015、1110ramp均与固定配置一致。U schema禁止真值且物理角色互斥；V eval/no_grad仅健康，无选模。

own final30实际config/预算/祖先/冻结严格加载；row自动exec到独立prediction和scorer，七view完整预检查后才读truth。每row126、共756指标，7056000预测。单owner、独占输出、计入其他进程的GPU容量和发布闭包/实际transport替换正确。

聚焦CPU与CUDA检查PASS，包含稳定门控、teacher无BN更新、CE无U前向、非法配置；全矩阵合成pipeline检查PASS，缺视图truthclosed。合成结果仅测试收尾，不能当真实成绩。真实checkpoint无querysmoke随remote preflight执行后直接继续。

## 单项技术修复

r01预检PASS，dispatch在启动前配置核对退出，独立日志/PID退出/run根仅空logs证实0worker。跨平台U_cache的str(Path)在Windows产生反斜杠，本地修复前回归失败。改为as_posix并将RUN/RELEASE/OWNER改新r02，原产物保留；算法/矩阵/seed/预算不变。针对性回归PASS，实际检查加入U_cache标准POSIX拼写。

唯一问题定点复审PASS：六row实际Windows配置与r02 spec完全一致，POSIX拼写与Linux一致；相对r01仅U_cache、run_id、output、launch_owner改变，矩阵/算法/seed/30epoch相同，不覆盖失败产物。
