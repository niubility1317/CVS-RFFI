# ADV3B02-FCR Task7报告：交叉重构、共享约束与latent cycle

## Status

Task7已完成本地模块级交叉损失组合和聚焦验证。实现范围仅为`phase1_fcr_losses.py`、新的交叉损失测试、FCR-10/11/12/16/17追踪行和本报告。没有接线训练循环、真实checkpoint、N607、target query或端到端性能声明；FCR-13继续保持`blocked`。

## Red/green evidence

1. 红测：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_cross_losses.py -v`。
   - 结果：收集时`ImportError: cannot import name 'compute_cross_losses'`，证明新接口在实现前不存在。
2. 初次绿测暴露测试夹具断言需修正：精确均值的高斯NLL仍保留常数项；原clean/LEO内容夹具相同，不能证明双侧共享梯度。测试改为正确目标严格优于反向目标，并使用不同共享内容。
3. 绿测：同一命令结果`5 passed`。
4. Task6回归、编译和空白检查结果见本任务提交前的最终验证记录。

## Public loss interface

`compute_cross_losses(...) -> FCRLossOutput`显式接收clean/LEO`FCRFactorOutput`、两套self decode、clean→LEO和LEO→clean decode、`FCRPairBatch`以及两条重新编码回调。它返回`total`、可微分`components`和仅诊断用的`metrics`。组件为`self`、`swap`、`swap_clean_to_leo`、`swap_leo_to_clean`、`shared`、`latent_cycle`、`eta`、`factor`和`anti_collapse`。

`self`和两条swap均复用Task6的有界异方差复IQ NLL；没有另一套NLL或MRSTFT实现。clean→LEO只与`pair.leo_iq`比较，LEO→clean只与`pair.clean_iq`比较。方向性掩码优先读取`pair_valid_mask["clean_to_leo"]`或`["leo_to_clean"]`，否则只能使用Task2同步`nuisance`掩码；无有效pair时返回连接至decode输出的精确有限零，不会随机补pair或除零。

## Shared、cycle和anti-collapse

`shared`逐token比较`z_s`，并对单位球归一化的`z_f_id`执行两项对称stop-gradient距离。因此clean和LEO各自都收到一致性梯度，没有单向teacher实现。

防塌缩项对联合clean/LEO内容和身份码分别施加std下限和off-diagonal covariance惩罚；常数码的std下限项严格为正，去相关且具有方差的表示更低。它不以无限扩大clean/LEO nuisance距离作为替代。

latent cycle实际把`clean_to_leo.mu_iq`和`leo_to_clean.mu_iq`送入相应回调重新编码。前向恢复detach后的clean`z_s/z_f_id`和LEO nuisance；反向恢复detach后的LEO`z_s/z_f_id`和clean nuisance。测试记录了两次回调调用、`[B,2,T]`合成输入和cycle项不向作为目标的原latent回传梯度。

## Eta/factor边界

`eta`只使用LEO结构化nuisance输出中的`eta_pred`与已知`pair.nuisance`对应字段；`nuisance_valid`可为逐字段mask，失效字段没有损失或梯度。函数不读取`pair.labels`、硬伪标签、`U_s`真TX或任何query。

`factor`包含clean和LEO各自的`z_s/z_f_id/z_n`cross-covariance；可选`domain_confusion_loss(z_f_id,receiver_id)`由训练调用方显式提供，适配条件域混淆或GRL路径。`probe_metric((clean_factors,leo_factors),receiver_id)`是训练外诊断接口，作为断开图的metric保存。当前模块没有将probe或类别标签混入训练损失。

## Trace、self-review和交接

仅FCR-10、FCR-11、FCR-12、FCR-16、FCR-17更新为本地模块证据支持的`implemented`；FCR-13未改动且保持`blocked`。自查确认：双向swap目的target和对应mask正确；shared为双向detach；cycle必经回编码并detach所有比较target；常数码罚项为正；eta无效字段严格零梯度；无效pair无随机fallback；没有读取标签、伪标签、`U_s`真值或query；没有重定义Task6 NLL/MRSTFT。

Tasks8/9/10可消费该函数，但训练接线必须保留当前Phase1数据权限和FCR-13能力边界，并在接线时提供真实方向mask、回编码网络、可选条件domain-confusion和独立probe。

## Commit and publish

本报告与Task7拥有文件同一提交，提交信息为`feat:add-FCR-cross-cycle-losses`。最终本地HEAD和远端OID独立一致性在提交后的任务回执记录；将同一提交OID写回报告会改变该提交对象，故不制造自引用OID。
