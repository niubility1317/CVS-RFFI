# Task4：OOF选择与全support refit分离

## 可追溯清单

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|T4-1|Required interfaces|`fit_sf_tapft`仅接受`best`与`final_step`，默认保持`best`|`code/cvsrffi/target_only_progressive_adapt.py`、`tests/test_target_only_progressive_adapt.py`|verified|完整26项测试通过|普通字符串及不可哈希非法值均由同一稳定分支拒绝|
|T4-2|Required interfaces|`final_step`只返回最后optimizer step状态并审计固定选择角色、精确step和完整训练行数|同上|verified|受控两步真实fit测试通过|score hook使step1赢`best`，`final_step`仍返回step2且状态不同|
|T4-3|Grouped opt-in|`full_support_refit=True`且OOF选择adapted时，使用fresh checkpoint、完整support和`selected_phase_steps`真实refit|同上|verified|真实3-fold+全support refit测试通过|`replace`固定统一schedule及top-k=1，调用不传inner validation|
|T4-4|Grouped result|全support结果独立于所有fold fitted result，`adapted_result`别名指向它，完整行数和`fold0_as_final=False`可审计|同上|verified|对象身份与训练行数断言通过|6行全support结果不等同于任一4行fold结果|
|T4-5|Zero-adapt|opt-in但OOF选择zero-adapt时不产生refit，训练数0且`fold0_as_final=False`|同上|verified|确定性zero-adapt测试通过|`adapted_result`与`full_support_result`均为`None`|
|T4-6|V1 compatibility|非opt-in grouped selection保留原`adapted_result=fitted_folds[0]`或`None`行为|同上|verified|确定性adapted legacy测试及既有selection测试通过|新增clean refit为显式opt-in|
|T4-7|Regression boundary|保持Task2 exact-state和Task3 stage schedule测试|同上|verified|完整26项测试通过|不写V2/query/R1/slimming|

## RED证据

- 首轮RED命令：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_target_only_progressive_adapt.py -q`。
- 首轮RED结果：`21 passed,5 failed`，退出码1；5项均为预期缺口：fit不接受`checkpoint_selection_mode`、grouped selection不接受`full_support_refit`且selection result没有`full_support_result`。
- 自审补充RED命令：同一文件加`-k rejects_unknown_checkpoint_selection_mode`；结果1 failed，证明不可哈希非法mode在set membership处错误抛出`TypeError`而非契约要求的`ValueError`。

## GREEN证据

- GREEN命令：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests/test_target_only_progressive_adapt.py -q`。
- GREEN结果：`26 passed`，退出码0；保留既有Torch2.1兼容分支1条`FutureWarning`，本任务未修改该分支。
- `git diff --check`退出码0；仅报告Git的LF→CRLF工作区提示，没有空白错误。

## 实现决策、自审与关注项

- API决策：`fit_sf_tapft(..., checkpoint_selection_mode="best")`保持V1默认；新增精确`final_step`模式。`select_sf_tapft_by_grouped_cv(..., full_support_refit=False)`保持V1默认；clean R0路径显式opt-in。
- 最后一步证明：测试用窄score hook让step1严格优于step2；`best.audit.selected_checkpoint_steps==(1,)`，而`final_step`固定为`(2,)`、角色为`fixed_final_step`，且返回模型/head状态与step1候选不同。
- 完整行消费证明：3-fold fixture每个fold fit审计4行；clean refit审计6行，与`len(target_train.physical_ids)`一致，`phase_steps==selected_phase_steps`，且full-support result不与任一fold result对象同一。
- 精确状态边界：最终单快照仍经过Task2许可delta恢复；非许可参数/buffer继续锚定，既有Sinc exact-state测试通过。OOF只决定adapted/zero-adapt和统一schedule，不平均fold参数。
- 反向可追溯审查：7项verified，0项deferred，0项rejected，0项blocked；属于Task4严格设计同构，不是近似实现。
- 交付提交：本报告、实现与测试由同一Task4提交固定；精确OID记录在任务返回中，避免提交正文自引用无法稳定的问题。
- 最高风险/顾虑：clean refit仍是显式opt-in，Task5尚未将R0 runner接入该路径；这是Task4要求的兼容边界，不是本任务遗漏。没有触碰`conversation_index/`、N607、query、source、V2、R1或slimming。
