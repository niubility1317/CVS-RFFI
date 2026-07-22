# D4b single-observation old-lock floor traceability

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D4B-01 | 用户目标；项目.md 7.1.1 | 每个scenario独立拟合，禁止跨scenario support拼接，固定接收IQ派生view不增加K或LEO状态 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | `pytest ...`:18 passed；cross-scenario lineage与K1回归 | 复用D4a lineage与固定接收IQ view原语 |
| D4B-02 | 用户目标 | after必须通过父`COMMIT.json`绑定并逐位复用同row before的每scenario旧类状态 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 父闭包、错误COMMIT拒绝、before→after runner集成单测 | 父状态成员必须只读、哈希和receipt匹配 |
| D4B-03 | 用户目标 | after的新类prototype只能由after注册support构建，旧support不得更新旧头 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 修改旧support后新prototype逐位不变；旧头逐位相等 | 旧support仅参与support-only floor guard |
| D4B-04 | 用户目标；项目.md 7.2 | query逐样本对全部注册类打分，无truth/role/quota/global assignment/query fit | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 无query拟合参数；batch扩展预测不变；receipt字段断言 | query仅执行不可变equalizer与联合cosine head |
| D4B-05 | 用户目标；项目.md 10.3 | scale/offset/门控只能由support LOO确定，并输出逐类floor与forgetting防护 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 每scenario逐旧/新类floor断言；旧类support forgetting不大于0 | 保护所有before正确旧support样本不受新类侵入 |
| D4B-06 | 用户目标；项目.md 10.3 | 极轻量：0epoch优先、参数不超过80k、持久状态不超过256KB、无dense query图 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | after参数0、0epoch、0 optimizer step、序列化状态上限单测 | 闭式注册，无optimizer step |
| D4B-07 | 可达路径 | runner候选、状态序列化、receipt、view lineage和parent closure完整可达 | `code/cvsrffi/stage2_diag_cosine_exploration.py`; `tests/test_stage2_diag_cosine_exploration.py` | verified | CLI解析与真实runner before→after集成；`pytest ...`:18 passed | 不提交Git，由主agent统一审阅提交 |

## Verification

```text
conda run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_diag_cosine_exploration.py code/cvsrffi/stage2_single_observation_floorlock.py
conda run -n ssr-gpu python -m pytest tests/test_stage2_diag_cosine_exploration.py tests/test_stage2_single_observation_floorlock.py -q
18 passed
```

Reverse audit：7项均为`verified`；无`deferred`、`rejected`或`blocked`。这是D4b设计的严格代码落地与本地合成测试验证，不是正式性能门槛证据；真实合法缓存上的开发row与独立确认矩阵仍需主流程执行。
