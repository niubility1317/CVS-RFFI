from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared
def score(row):return shared.score(row,design=d,prediction_preflight=preflight)
def finish():
 result=shared.finish(design=d)
 path=d.BASE/'final_analysis.md'
 with path.open('a',encoding='utf-8') as f:
  f.write('\n本轮 33 标签专用轮、17 U 激活轮是曝光量计数。TAIL=34..50，MID=17..33，SPREAD=17/19/..49；三组 L 每轮参与。日历与学习率在 target 访问前固定。当前教师源 V 代理不能证明隐藏 U 精度；真实 AdamW 双临时分支按原 moments/clip5 计算一步 U 增量，真实模型仍每步只更新一次。\n')
 return result
