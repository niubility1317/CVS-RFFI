# 全候选目标域历史复测

用户已明确授权全部14候选目标域测试及详细数据。冻结source run为core90_anchored_s392005_20260911_r1，seed392005；候选A0/A1/A2/A3/A4/C_angle/C_angle_keep/P1/A5-0/25/50/75/100/A6，P1主表使用full。A0为H0对照。

只做预测与独立评分，不训练、不适应、不改变温度/拒识/门控，不按target结果选模或选择性重跑。此为历史目标集描述性复测，不能声明独立确认或科学晋级。上次仅口头确认测试但没有启动，本run首次实际执行该授权。

数据契约沿用core90_evidence_frozen_392005_20260911/data_contract.json；原ManySig equalized=1、out_len=256、center crop、RMS normalize、center=false。目标RX=0/2/5/7/9/10/11、day=0–3、六TX、168000物理记录；每个场景都覆盖相同完整ID集合。clean与三个leo_*_weak各自测试，沿用已有sha256(seed,scene,opaque ID)逐记录CPU信道生成规则。此次全场景压力复测不代表单观测独立确认。

冻结H0来源为原run/H0/ground_checkpoint.pt，已经核实scratch=true、upstream=[]、final_only、target_feedback=false且数据契约一致。加载每个bundle后检查实际H0权重和TX映射相同，保留source V温度/阈值。新入口仅接通已冻结系统，实际方法不变。

执行使用N607普通身份及CVS-RFFI Python。GPU0/1/2/3各一个scene worker，GPU4现有任务不动。输出runs/core90_anchored_target_all_s392005_20260912_r1，完整日志位于同名logs目录。唯一launch owner为当前主Agent，目录排他创建；各scene先真实source IQ/H0 smoke，再预测全部14候选。全部56个预测文件封存后，新的独立scorer进程才按opaque ID连接truth，输出overall、RX、TX、day、RX×TX、confusion与拒识覆盖率。

所有输入从既有物理ID按原预处理读取，IQ-only边界丢弃标签和元数据；预测只接受IQ/冻结特征。技术错误保留产物，不自动重试、不评分不完整预测、不干预无关进程。

## 本地检查与实际启动：VERIFIED

本地ssr-gpu解释器C:/Users/lh594/.conda/envs/ssr-gpu/python.exe实际导入入口与scorer正确/接受计数测试通过；独立P0/P1审查未发现阻断项。实际代码18befc4b2c1d0d2cfcc1936af664678d8c29dc26，GitHub远端OID读回相同。归档一次本地/远端SHA比较通过，远端一次compileall通过。

release=/home/szu2070436088/2510044040/CV-SincNet/releases/anchored_target_18befc4b。协调器PID3240390；四worker PID3240402/3240405/3240408/3240411，PPID均为3240390，对应GPU0/1/2/3，每个显存614MiB。实际argv包含对应scene、source、ground、contract和本run输出，CWD读回release一致。四worker均输出REAL_CHECKPOINT_SOURCE_SMOKE_PASS并进入真实target预测。原GPU4任务PID612456未变。

## 完成与交付

状态VERIFIED/SCORED，全部14候选×4场景完成。56份预测封存后独立评分，独立复核与分组求和检查通过，无运行错误。详见[detailed_target_results.md](detailed_target_results.md)。
