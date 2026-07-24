# C-DOM-SCXMAP-D92-GLF/r1 Phase1留出证伪r2报告

根目录权威报告镜像：`E:\type10-7\automation_reports\CV-SincNet\scxmap_p1_held_falsifier_r2_20260724\report.md`。

r1在N607 Torch`2.1.0+cu121`的real-checkpoint support-only smoke因不存在公开`safe_globals`而技术失败，54行未开始，永久为`NO_PERFORMANCE_RESULT`。r2保持SCXMAP方法、54行矩阵、晋级门和Target25硬禁完全不变；唯一修复是新增`safe_checkpoint_state.py`，在本地用公开allowlist＋`weights_only=True`把原checkpoint转换为来源SHA绑定的纯`model/ema_model` tensor artifact，再由N607 Torch2.1原生`weights_only=True`验证。

纯tensor checkpoint SHA256=`e027eca9e717e231a9d548bc2e6f2fc56829a86cfd5c5e2332241cbe559b842d`，receipt文件SHA256=`281b76b704b9c012da32dd3be5bd90ed9b747391d826fe44075909eb2f7369df`，state digest=`9c0cda23c63f9777da248166397fd21dc112978c20db758e527ebe0c6c678816`，390个tensor、195个model key、2,100,078个model＋EMA参数。当前`py_compile`和安全checkpoint＋SCXMAP＋transform＋R2A回归共`19 passed`，真实support-only smoke为`PASS`；待独立审查终审、Git冻结和新run发布。禁止`weights_only=False`、私有allowlist、续跑r1、打开Target25或GitHub上传。
