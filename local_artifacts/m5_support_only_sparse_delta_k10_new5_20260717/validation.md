# M5验证记录

- `ssr-gpu`下`python -m py_compile run_m5.py`：PASS。
- 完整训练日志：90行，等于3候选×3场景×2注册状态×5step。
- 精确白名单参数量：A=25,760、B=22,080、C=25,760；均小于50,000。
- 每次适配5epoch/5step；SGD momentum=0；未保存optimizer状态。
- 18个FP16差分patch均已落盘；未保存全模型副本。
- 三个prediction artifact schema均严格为`query_token`、`before_predicted_class_index`、`after_predicted_class_index`，不含truth/role/quota。
- `selector_lock.json`确认`query_opened_for_selection=false`；三场景使用相同白名单与超参数。
- 所有候选/场景/注册状态的`patch+head<=256KB`、`deployment_added_macs_after_merge=0`、epoch/step上限：PASS。
- A/C未打开query；预锁定B只执行一次formal query隔离测试。truth sidecar在全部三场景prediction artifact封存后才由scorer打开，query结果不反馈任何训练、适配、校准、选择、早停、回滚、排名或调参。
- 外层首次运行曾在64秒命令包装超时后被核验为已经完整产出；修正严格scorer时序后以180秒上限完整重跑，80.1秒正常退出，结果确定性一致。

## SHA256

- `run_m5.py`：`2D64E8EB20AA5FD5D9E8612C507CF7DC3C27CE4FE02BF0A6AB3C8D10F3A52C85`
- `results.json`：`95771CE9407251C6D37B67DD565F54FA62831162897DE76281B6FDD6815A4F74`
- `selector_lock.json`：`5ECA56E6C1D9770563DFF341AC11077DB3CC604EACE66F085B4531AD411568F8`
- `training_log.jsonl`：`05BEE951EC52EE1E5B90CDBD66D4F4C7A8CB90D01D0408CA87BF7D366EB80C51`
