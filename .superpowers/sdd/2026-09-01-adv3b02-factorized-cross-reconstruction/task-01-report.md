# ADV3B02-FCR Task1报告

- Status:`DONE`

## 修改文件

- `code/cvsrffi/phase1_fcr_types.py`：新增Task1共享FCR数据类型与精确默认值。
- `code/model_dual_cvsincnet.py`：为模型构造函数和`build_dual_model(...)`增加关闭态兼容的`use_fcr`/`fcr_config`参数；Task1只保留配置，不创建FCR模块。
- `code/tests/test_phase1_fcr_model_contract.py`：关闭态严格加载和逐元素输出兼容测试。
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`：仅将FCR-25更新为`verified`。

## TDD证据

- 红测命令：`conda run -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_model_contract.py -q`
- 红测结果：预期失败，`TypeError: build_dual_model() got an unexpected keyword argument 'use_fcr'`。
- 绿测命令：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_model_contract.py -q`
- 绿测结果：`1 passed`；仅有既有`torch.cuda.amp.autocast`弃用警告。
- 编译命令：`conda run --no-capture-output -n ssr-gpu python -m py_compile code/model_dual_cvsincnet.py code/cvsrffi/phase1_fcr_types.py`
- 编译结果：exit 0。

## 兼容性与自检

- `use_fcr=False`时`self.use_fcr=False`、`self.fcr_config=None`、`self.fcr=None`。
- legacy state dict以`strict=True`加载成功，state dict不含`fcr.*`键；相同seed和输入下`tx_logits`、`z_id`逐元素一致。
- `use_fcr=True`仅保留传入的`FCRConfig`，`self.fcr`仍为`None`，未提前创建未实现模块。
- 已运行`git diff --check`；限定diff只涉及本任务拥有的四个交付文件。工作树未发现其他待提交修改。

## Git闭合

- Commit OID：由包含本报告的`feat:add-FCR-contract`提交在创建后独立读取；由于Git对象ID由本报告内容本身决定，权威OID记录在本任务最终交接中。
- Push结果：由提交后`git push --set-upstream origin HEAD`或当前上游推送命令独立读取。
- Remote OID readback：由提交后`git ls-remote origin refs/heads/codex/adv3b02-fcr-20260901`与本地`HEAD`比较后记录在最终交接中。

## 关注点和后续接口说明

- Task1不实例化FCR网络或训练路径；后续任务须在显式`use_fcr=True`时实现模块，并保持本关闭态测试不变。
