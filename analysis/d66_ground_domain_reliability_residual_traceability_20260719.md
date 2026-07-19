# D66地面域可靠性残差追踪

## 预注册背景

D62是当前聚合最强开发点：注册前旧类92.78%、注册后旧类82.22%、seen-new 84.67%、H 82.62%、遗忘10.56pp、joint 26.67%、min-before/min-after/min-new为80.00%/53.33%/73.33%，混淆old→new/new→old/new→wrong-new为23/8/15。D65通过冻结Stage2-B旧类几何把注册后旧类提高到86.11%、遗忘降到6.11pp，但seen-new降到59.33%；地面int8组件只做只读审计，实际拟合输入计数为0。

历史排除证据如下：D19强弱旧类anchor使seen-new降至6.67%–12%；D25直接ground-z旧中心融合仅50.00% after-old、42.00% seen-new，逐块半径似然造成46.11pp遗忘；D30的DALI虽在45/45外折实际参与，但没有改变任何预测；D36按旧类专属ground anchor、offset和IRLS只交换错误，最优联合结果仍为66.11% after-old、52.00% seen-new。旧anchor中心替换、旧类半径似然、old-old重排、角色截距、Procrustes/transport和query batch统计均不得在D66复现。

## 唯一路线与公式锁

D66名称为`ground_domain_reliability_residual`。它不把地面原型当作旧类最终中心，而只从不可变Phase1 int8域×类聚合组件估计z160坐标的域稳定性。设反量化聚合原型为`p[d,c,j]`，mask为`m[d,c]`：

- `mu[c,j]=mean_d p[d,c,j]`；
- `W[j]=mean_(d,c) (p[d,c,j]-mu[c,j])^2`，表示同类跨地面域漂移；
- `B[j]=mean_c (mu[c,j]-mean_c mu[c,j])^2`，表示类间身份信号；
- `r[j]=(B[j]+eps)/(B[j]+W[j]+2eps)`；
- `s[j]=sqrt(1+r[j])`。

`eps`固定为float32机器精度。由`0<r<1`，`1<s<sqrt(2)`；不设置alpha、rank、阈值、温度、场景/接收机参数或扫描。最终288维共享变换为`S=diag(s[0:160],1[160:288])`。D62的全部full/block、outer/inner支持拟合都在`x'=xS`坐标执行，随后把单一仿射状态编译回原坐标：`W= W' S`，截距不变。Stage2-B旧类、Stage2-C所有已注册旧/新类、held support和未来逐条query均使用同一公式；query只读取编译后的int8/FP16仿射状态，额外MAC和持久状态均为0。

地面组件只在适配入口瞬时反量化并计算160个共享尺度；不持久化full-precision ground bank，不更新组件，不读取原始IQ、样本级feature、clean/source样本或query。D66必须记录组件入口/出口SHA、84个有效cell、类/域数、尺度SHA及统计。当前组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，故本轮只能进行用户已授权的开发support内部held-rank探针，不获得formal/query/125声明资格。

## 实验锁与判门

- cell：receiver`20-1`、seed`713101`、K10/new5、clear/low_elev/rain三场景×5 outer fold；实际每折K8。
- 数据：复用匹配capsule/split和`p2_min_v1`的`VALIDATED_ONCE`D18 enrollment-only support，不因方法变化重验数据。
- 候选矩阵：沿用D42七候选105行；仅D42 INT8/FP32匹配行替换为D66机制，其余对照原样保留。
- 协议：单物理IQ单一LEO_weak observation；query、clean/source、truth/role Oracle、class quota、全局重分配均不可达。
- 量化：FP32/int8 before/final argmax变化和margin符号翻转必须全为0，所有分数有限。
- 性能：相对D62必须保持总体before/after/new/H/joint、三项class floor、三场景同类指标、遗忘和三类混淆均不退化，并至少严格改善after、forgetting、joint或任一floor；否则停止D66。
- 资源：地面统计计算和所有D62 fit完整计入适配MAC/临时内存；query额外MAC/state固定为0。

即使开发门通过，本轮也不直接运行第二seed或125。D66完成是D64–D66连续第三轮，必须先执行强制回顾，再决定下一路线。

## 待实现与验证

- 新增独立D66 probe和专项测试，不修改D62历史源码与artifact。
- 验证mask反量化、公式边界、类置换不变、组件只读、对全类统一变换、系数编译等价、K1/K2精确兼容、资源闭包与105行输出闭包。
- 在干净worktree运行D43–D66完整相邻测试链，再启动真实105行开发实验。

## 合成集成烟测归因复核

初版专项4/4、D42–D66完整回归303/303通过。额外随机合成的真实D42＋D62烟测在内部inner fit触发`D42 sklearn coefficient deployment prediction drift`，但移除D66并用未改动D62在同一数据上复跑后出现完全相同的栈和错误；因此该失败来自合成集本身触发D62既有近边界fail-closed条件，不能归因于地面共享尺度，也不能作为改变预注册机制的证据。D66保留原始“全部D62拟合在共享坐标内执行、随后编译回原坐标”公式；不放宽任何D42断言。真实集成判据为锁定项目enrollment-only support上的105行fail-closed运行。

## 实现与本地验证

- 实现：`code/scripts/probe_d66_ground_domain_reliability_residual.py`。
- 测试：`tests/test_probe_d66_ground_domain_reliability_residual.py`。
- 真实组件审计：26域、6类、84个有效cell，每类14个；`r`范围0.0242749–0.9999186、均值0.7698918；`s`范围1.0120647–1.4141848、均值1.3240636、条件数1.3973265、SHA256=`70a8e94327e7100695f691d6ae49e246305036cefd92579e977e3d536c37df6c`；FFT/RF尺度逐bit为1。
- 资源静态值：地面组件逻辑状态25,428B、瞬时反量化53,760B、一次可靠性统计58,880标量MAC；K8的before6类＋final11类共享变换应用21,760MAC，query额外MAC/state均为0。
- `py_compile`通过；D66专项4/4通过；D42–D66完整26文件303/303通过，用时81.1s；`git diff --check`通过。

## Resource-R1审计修复

首次真实运行完成105/105行、query0、性能与组件使用字段完整，但全日志解析发现runner在D66 top-level资源更新后又以编译仿射头大小覆盖`persistent_state_bytes`：行内同时记录组件逻辑25,428B和实际输入84，却把总状态写成仅头部8,583B。正确总状态应为8,583＋25,428=34,011B。该问题只影响资源主字段，不影响拟合、预测、量化或性能，但当前artifact不能作为最终D66资源证据。

Resource-R1在D66 probe内包装锁定runner的D42 fold结果：先保留`d66_compiled_affine_state_bytes`，再写入`d66_component_inclusive_persistent_state_bytes=head+component`并同步主`persistent_state_bytes`；验证器新增严格加总等式。候选、公式、数据、性能门和所有预测路径不变。首次输出原样保留；修复运行使用新目录，不覆盖旧artifact。
