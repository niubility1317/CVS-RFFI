from pathlib import Path
import json
P=Path(__file__).parent
def write(name,v):(P/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
m=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
m['seed_roles'].update(augmentation=340299,batch='L generator=299+epoch; independent U generator=1299+epoch')
write('manifest.json',m)
s=json.loads((P/'registry_spec.json').read_text(encoding='utf-8'))
s['tags']=['STAR','RIEI','Loo','ordinary-pseudo-label','warm10','no-unlabeled']
s['comparison_group_id']='riei-simple-warm10-pl-vs-no-u'
s['authorization']='User requested simple RIEI optimization: moderate SG first10 epochs, E100, PL from E20 vs no unlabeled samples; prior explicit authorization to exceed two GPU jobs.'
s['metrics_plan']['scorer_ref']=str(P/'train.py')+'::score'
s['data']['roles']['U_s']='.90 of source pool; loaded only in PL row, used only E20+; noU row never indexes U IQ'
for row in s['rows']:
    row['seeds']['augmentation']=340299
    row['seed_notes']='Model299; stratified split299+rx; fixed warm augmentation340299; L shuffle299+epoch, independent U shuffle1299+epoch; original full-SG fixed arrays.'
write('registry_spec.json',s)
write('review.json',dict(status='PASS_NO_P0_P1',reviewer='/root/review_baselines',scope='warm raw256/time FFT layout, noU data access, PL20, matched L order, E100 dispatch',metadata_note='seed roles corrected after independent review'))
report='''# RIEI简单暖启动：100轮、有/无伪标签对照

状态：PREPARED。两组均从零训练，seed299；固定E100 student为主结果。未加载历史checkpoint。

## 简要诊断

已完整解析旧r03的200轮、5800批日志，并从冻结概率独立重算最终准确率：source clean88.00%、source SG22.00%、target clean57.375%、target SG17.50%。stdout无异常。训练SG CE在E100–120均值仅0.0746，但同期source SG仅23.5–25%，支持固定强衰落训练样本拟合较好、SG泛化差的判断，不能证明唯一原因。

原信道是逐采样点Loo，log-amplitude标准差3（delta2=9）；从第一轮施加强扰动可能增加早期学习困难。原PL直到E100才启用，且SG视图接受量低于clean。这里只做用户指定的温和起步与提前PL，不据target分数选择checkpoint或调参。

## 两组配置

|组|训练轮数|无标签数据|伪标签|
|---|---:|---|---|
|noU|100|不索引U的IQ，不构造U loader，不做U前向|始终关闭|
|PL20|100|仅source U，E20起前向|E20含当轮启用，阈值0.9，系数0.65|

两组E1–10使用固定温和Loo：delta2=1，即log-amplitude标准差1；散射方差0.0049、逐点衰落、逐原始256段RMS归一化均保持原实现。E11–100恢复原固定强度SG。测试全程使用原强度固定数据，未弱化测试。温和增强seed340299；两组同一L顺序（299+epoch），U独立生成器（1299+epoch）。

RIEI原优化版保持不变：clean CE与SG CE系数1:1，RX CE系数1、MI系数1.2、IE系数-1.2、feature norm系数0.0001。CE/MI/IE仍sum、norm仍mean；Adam1e-4常数、L batch64、U batch256，交替更新heads/FED。无STAR E调度、动态增强、EMA或额外CFO/IQ。

输入先在原始两段256信号施加信道，再补零至512并分别FFT，最终布局time1/time2/freq1/freq2。不向已拼接FFT特征直接施加信道。

## 验证与证据

verification.json：本地CUDA真实RIEI更新通过；E19/E20的PL边界、无U步骤、禁止U IQ访问的loader守卫测试、暖增强布局/确定性及语法检查均通过。review.json：独立P0/P1审查无阻断问题。

diagnosis.json保存旧实验完整日志汇总与独立重算。新结果须等待各自COMPLETE.json及最终冻结预测独立评分；当前不宣称提升。本实验是目标结果已知后的单seed探索，不能作为盲测或显著性证据。
'''
(P/'report.md').write_text(report,encoding='utf-8')
print('METADATA_READY')
