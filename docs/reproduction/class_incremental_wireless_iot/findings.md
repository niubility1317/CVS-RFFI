# Findings

## Control Files

- `E:\type10-7\AGENTS.md` requires reading `项目.md` before CVS-RFFI/CV-SincNet research, experiment, automation, paper, or Stage2 interpretation work.
- `E:\type10-7\项目.md` defines CVS as weak-label/semi-supervised source-domain DG plus spaceborne few-shot adaptation. It requires `R_s`/`R_t` disjointness, explicit `Y_old`/`Y_new`/`Y_unknown`, positive `K`, and separation of Phase2 old/new learning from Phase3 unknown rejection.
- Root `E:\type10-7` and `E:\type10-7\paper_reproduction` are not Git repositories. `E:\type10-7\github_publish\CVS-RFFI-repo` is the Git-backed release surface, currently with unrelated dirty changes.

## Paper Extraction

- PDF共9页，可由`pdfplumber 0.11.9`抽取；主线程只保留人工整理后的页码证据和对照清单，未把全文或结构化摘录artifact纳入发布目录。
- 论文任务是ADS-B NDI class-incremental learning，不是WiSig/CVS Stage2。忠实复现必须保留ADS-B、100类、5阶段、无历史数据、CSIL通道分离、zero-bias cosine fingerprints、KD+EWC和DoC。
- PDF明确给出USRP B210、`1090 MHz`、`8 MHz`、每条消息前`1024`个complex samples、`32x32x3`输入、`60/40`训练验证、`100`个transponders、每批`20`类、增量训练batch size `64`、`10`epochs、SGD、lr `0.01`、momentum `0.9`、L2 `0.01`。
- PDF缺失具体100类ID、stage class list、seed、stage0 epochs、KD/EWC权重、FI估计细节、完整Conv2d padding/activation/pooling细节和曲线原始点。

## Repo Mapping

- Git承载面中已有`paper_reproduction/protonet_cda`、`feature_separation_crossrx`、`receiver_agnostic_twostage_uda`、`mitigating_receiver_impact_da`和`cvs_aligned`。
- 本论文应新增`paper_reproduction/csil_class_incremental_iot/`，不能放入`code/cvsrffi`主线。
- 可复用`paper_reproduction.common.config`的配置加载和formal placeholder检查；CVS扩展若后续需要，必须另放`cvs_extension=true`路径。

## Protocol Mapping

- Paper-faithful层只声明`paper_faithful_adsb_class_incremental_only`。
- CVS扩展层若做，必须重新定义`R_s/R_t`、`Y_old/Y_new/Y_unknown`、`K`和satellite/LEO view；否则只能标为`NON_LAUNCH_DIAGNOSTIC`。
- 本轮未SSH、未SCP、未启动N607实验。

## Audit Checklist

- 独立审计要求逐条覆盖任务设定、输入信号、标签协议、数据划分、特征构造、模型主干、zero-bias层、DoC、CSIL结构、增量训练损失、参数锁定、增量批次、训练超参、baselines、upper bound、Fig.7/8/9、TableI/II和代码不可见假设。
- 当前实现完成协议校验、CSIL分类头/通道扩展、损失、DoC/accuracy指标、dry-run边界和单元测试；真实ADS-B训练、baseline矩阵、消融矩阵和图表复现实验仍未完成。
