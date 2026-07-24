# C-DOM-SCXMAP-D92-GLF/r1 Phase1留出证伪r2报告

根目录权威报告镜像：`E:\type10-7\automation_reports\CV-SincNet\scxmap_p1_held_falsifier_r2_20260724\report.md`。

r1在N607 Torch`2.1.0+cu121`的real-checkpoint support-only smoke因不存在公开`safe_globals`而技术失败，54行未开始，永久为`NO_PERFORMANCE_RESULT`。r2保持SCXMAP方法、54行矩阵、晋级门和Target25硬禁完全不变；唯一修复是新增`safe_checkpoint_state.py`，在本地用公开allowlist＋`weights_only=True`把原checkpoint转换为来源SHA绑定的纯`model/ema_model` tensor artifact，再由N607 Torch2.1原生`weights_only=True`验证。

纯tensor checkpoint SHA256=`e027eca9e717e231a9d548bc2e6f2fc56829a86cfd5c5e2332241cbe559b842d`，receipt文件SHA256=`281b76b704b9c012da32dd3be5bd90ed9b747391d826fe44075909eb2f7369df`，state digest=`9c0cda23c63f9777da248166397fd21dc112978c20db758e527ebe0c6c678816`，390个tensor、195个model key、2,100,078个model＋EMA参数。`py_compile`和安全checkpoint＋SCXMAP＋transform＋R2A回归共`19 passed`，真实support-only smoke为`PASS`；独立代码终审`P0=0、P1=0、P2=0`。冻结发布源码commit=`d48eed24521cd6848b6d0fa8f40dd876a45192fd`；`source_d48eed24.zip`为35928383B、SHA256=`7e2ec4022c4c3e023a75c59580881b8f15e3a052c03dd815be1aa5108ad27545`；15项`release_receipt.json`文件SHA256=`a4919a4173972d2438b3a65268e5ee97dcf01dee7cf43eb26ba9e745c901c3f8`；归档入口SHA256=`73ec2a2d976781d786ae18f9d6fab21c29dc319034545b2dbbdecf3ba6c0df22`且`bash -n`通过。独立release-surface终审`P0=0、P1=0、P2=1`，批准仅Phase1-held proxy landing，Target25仍为`false`。唯一P2仅为字节口径说明：ZIP文件集合与`d48eed24521cd6848b6d0fa8f40dd876a45192fd`Git树完全一致，3249个文本entry统一CRLF→LF后与Git blob精确一致；N607与receipt审计以ZIP原始entry SHA为准，无需重建archive/receipt。纯tensor artifact不进入Git/GitHub。禁止`weights_only=False`、私有allowlist、续跑r1、打开Target25或GitHub上传。

## N607终态与结果

2026-07-24 17:38:40 CST在物理GPU0启动，PID=`2003887`，17:38:55.949 CST健康完成；`pipeline.exit=0`，54/54行、prediction=1、score=1、output=9，远端/本地SHA闭合。Torch`2.1.0+cu121`仅用`weights_only=True`；`query_rows_used_for_fit=0`、`optimizer_steps=0`、`formal_phase2_eligible=false`、`bundle_created=false`、`target25_release_authorized=false`。最终本run PID为0、GPU和SSH均清理；未重试、未覆盖、未续跑。

|分组|臂|old-before|old-after|seen-new|H|floor|F|Δold|Δnew|ΔH|Gate|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Overall|M0|84.6894|82.2143|82.3102|80.8179|58.6433|2.4751|—|—|—|FAIL|
|Overall|M_DA|84.6446|82.1803|82.2833|80.7858|58.6433|2.4643|-0.0340|-0.0268|-0.0320|FAIL|
|K1|M0|81.3607|78.2163|78.6009|75.5726|44.2681|3.1443|—|—|—|FAIL|
|K1|M_DA|81.2261|78.1143|78.5204|75.4764|44.2681|3.1118|-0.1020|-0.0805|-0.0961|FAIL|
|K5|M0/M_DA|86.8572|84.7647|84.6920|83.9937|65.8482|2.0925|0|0|0|FAIL|
|K10|M0/M_DA|85.8505|83.6620|83.6377|82.8873|65.8135|2.1885|0|0|0|FAIL|

`proxy_gate_pass=false`：9个K×scene与18个K×pseudo-new分层全部失败。48/54行beta非零，17,580/19,782条注册后query的margin变化，但K5/K10的argmax变化为0；K1仅8次变化，wrong→correct=0、correct→wrong=7。持久状态4,873–4,909B；fit/query matrix MAC总计1,866,240/25,637,472；优化步和query-fit行均为0。结论是完整、合法的Phase1-held proxy阴性证据：淘汰SCXMAP，不创建bundle，不开放Target25，不做fresh retry。完整权威报告与14个回收文件保存在根目录同run路径。

## 三轮回顾

已按`AGENTS.md`在第四轮前重读当前目标、`项目.md`、项目会话索引及SVRN/ADV3B02/SCXMAP完整日志与结构化结果。三轮均为技术完整、科学阴性：SVRN完整125相对D62的old-after/new/H下降`21.36/35.64/31.84pp`；ADV3B02完整125的M_DA仅Δold/new/H=`+0.0378/-0.0460/-0.0272pp`且`I_syn(H)=0`；SCXMAP虽48/54行beta非零、17,580条margin变化，但K5/K10决策不变、K1只有破坏。下一候选必须直接证明neighbor/argmax净纠错、old/new同权收益和全注册类floor提升；保留D92＋原始qKNN matched control，禁止继续调SCXMAP beta/rank或重演共同变换。Target25固定seed=`713102`，只有新候选通过Phase1-held和独立审查后才能发布。当前未冻结下一方法、未授权第四轮N607实验。
