"""Mirror authored report, curated tables and configs; retain raw runtime evidence locally."""
from pathlib import Path
import json,re,shutil
B=Path('E:/type10-7');S=B/'automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/report_bundle';W=B/'code/snapshots/daot_game_analysis_20260914_wt';D=W/'docs/research/daot_fasttrust_game_20260914'
D.mkdir(parents=True,exist_ok=True);copied=[]
def clean(s):
 s=re.sub(r'/home/szu\d+/', '/home/RESEARCH_USER/',s)
 s=s.replace('RESEARCH_USER','RESEARCH_USER')
 return s
for p in S.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(S)
 if rel.parts[0] in ['evidence','code_snapshot']:continue
 if rel.parts[0]=='sources' and p.suffix=='.json':continue
 q=D/rel;q.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix in ['.md','.csv','.json','.py','.svg']:
  s=clean(p.read_text(encoding='utf-8-sig'))
  if p.suffix=='.json' and 'A1_CAPTURED' in rel.parts:
   d=json.loads(s);d.pop('state',None);s=json.dumps(d,ensure_ascii=False,indent=2)
  if rel.as_posix()=='report.md':s=s.replace('](code_snapshot/','](../../../experiments/adv3b02_xuc/code/')
  q.write_text(s,encoding='utf-8')
 else:shutil.copy2(p,q)
 copied.append(rel.as_posix())
(D/'DELIVERY_SCOPE.md').write_text('# 交付范围\n\n本目录包含完整分析正文、配置阅读附录、逐行配置、数值明细与图。原始远端日志/进程状态证据和训练代码副本保留在本机完整报告包，未重复纳入Git；正文源码链接指向本分支对应的既有冻结实现。机器账户名称已在发布副本中替换。source_manifest记录本机完整包的源索引，因此包含仅在本机完整包保留的证据项。没有修改训练代码、实验参数或远端任务。\n',encoding='utf-8')
broken=[]
for f in [D/'report.md',D/'configuration_appendix.md']:
 for link in re.findall(r'\]\(([^)]+)\)',f.read_text(encoding='utf-8')):
  if re.match(r'^[a-zA-Z]+:|^#',link):continue
  if not (f.parent/link.split('#')[0]).exists():broken.append(link)
assert not broken,broken
print(json.dumps({'files':len(copied)+1,'bytes':sum(p.stat().st_size for p in D.rglob('*')if p.is_file()),'main_links_verified':True}))
