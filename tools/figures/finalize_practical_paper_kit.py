"""Publish the verified paper kit and mark obsolete local projections clearly."""
import csv
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path('E:/type10-7')
OUT=ROOT/'exports/unified_figures/20261024-practical-residual-paper-r03'
RUN='20261023-phase1-practical-residual-full2x2-m600-r01'
DOC=ROOT/'docs/project_governance/20261024_comparison_retirement'
PUBLIC=ROOT/'automation_reports/CV-SincNet'/RUN/'paper_assets_20261024'
OLD_RUNS=['20261019-phase1-external-dg-concat-pl-m219-r01','20261019-phase1-external-dg-concat-pl-m219-r02',
 '20261019-phase1-external-dg-concat-pl-m219-r03','20261021-phase1-dg-sourcecount1to6-m432-r01',
 '20261022-phase1-DURA-external-Lonly-readouts-m24-r01','20261023-phase1-external-DURA-readouts-m492-r01',
 '20261023-phase1-external-DURA-analysis-m516-r01']

def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((OUT/'source_data/summary_overall.csv').open(encoding='utf-8',newline='')))
    src={(r['protocol'],r['family'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in rows}
    checked=0
    for p in (OUT/'tables').glob('Fig*.csv'):
        for r in csv.DictReader(p.open(encoding='utf-8',newline='')):
            k=tuple(r[x] for x in ('protocol','family','cell','test','readout','view','metric'))
            assert r==src[k],(p,k);checked+=1
    def val(p,f,c,t='cross_rx',rd=None,v='six_practical_mean'):
        return src[p,f,c,t,rd or ('DURA-LU' if c in ('B','D','native_T1') else 'DURA-L'),v,'accuracy']
    def fmt(r):return f"{float(r['mean']):.2f} ± {float(r['sd']):.2f}"
    lines=['# 论文写作要点：使用新 practical residual 数据','',
      '全部数据来源于 '+RUN+'；627 行完整读出，四外部方法全部为 Full。以下固定展示 D（增强＋普通 PL），不是按目标结果挑选的最佳组合。PAMF 监督版为 C；原生半监督为固定 T1。','',
      '## 主要结果','', '|方法|跨日 S6|跨接收机 S6|双跨 S6|','|---|---:|---:|---:|']
    for f,c,n in [('riei','D','RIEI'),('drift','D','DRIFT-v2'),('ibrff','D','IB-RFF'),('sdg','D','SDG'),('pamf','C','PAMF 监督'),('pamf','native_T1','PAMF 原生半监督')]:
        lines.append('|'+n+'|'+'|'.join(fmt(val('F0',f,c,t))+'%' for t in ('cross_date','cross_rx','cross_both'))+'|')
    t1=val('F0','pamf','native_T1');direct=val('F0','pamf','native_T1',rd='raw')
    lines += ['',f"F0 跨接收机的原生 PAMF 从直接识别的 {fmt(direct)}% 提升至恢复后的 {fmt(t1)}%。这是两端实际均值；配对增益标准差另见 restoration_gain_overall.csv，不能用两个端点 SD 相减。",
      '', '## 需要保留的负结果与边界','',
      '- 原生半监督在 F2 双跨为 '+fmt(val('F2_rx_repartition_r01','pamf','native_T1','cross_both'))+'%，监督 C 为 '+fmt(val('F2_rx_repartition_r01','pamf','C','cross_both'))+'%；不能宣称半监督在所有划分优于监督。',
      '- 原始 F 的增强效应在无 PL 的 36/132 个条件、带 PL 的 30/132 个条件中均值非正；PL 效应在无增强的 69/132、有增强的 64/132 个条件中均值非正。条件是 4 方法×11 契约×3 任务，不是独立重复试验。',
      '- 恢复后 A/C 用 G-L，B/D/T1 用 G-LU，跨 PL 对比还改变了恢复器的 U 权限。只有监督 F＋G-L 是完整 L-only 系统。',
      '- 61 个外部 PL 训练行接受数为 0；完整原生门控接受率和部分实际增强计数缺失，不补造。',
      '- 三 F 种子条件于固定 G；SD 不是显著性检验。训练日期是 D1–D3，D0 为较早未见日。ManyRx 使用十类，不并入六类 ManySig 总体。',
      '', '## 论文组织建议','',
      '主文按“设置与三类泛化→Full 方法比较及对应 PAMF 监督/半监督→标注比例与源数量→PAMF 机制/结构→效率”排列。A/B/C/D 表单列，取消无关外部 CE-only/pair 大表。PAMF 的监督表征学习与原生半监督学习分别解释；结构、CRIC/一致性与无标签机制不能按目标收益重新归组。',
      '', '## 文件入口','',
      '- manuscript_results_en.md：可直接改写进论文的英文段落。',
      '- figure_captions_en.md：22 图英文图注。',
      '- tables/paper_tables.tex：三种泛化的紧凑 LaTeX 表。',
      '- tables/all_627_clean_S6.csv：所有组合的 Accuracy/Macro-F1。',
      '- tables/paper_main_comparison.csv：11 契约的直接/恢复 clean/S6。',
      '- historical_PAMF：有效原机制与时延图，原 168000 包口径，不能伪装成新 126000 包端点。',
      '- IEEE_experiments/main.pdf：6 页实验部分，使用新独立图，表距与浮动间距已检查。', '']
    (OUT/'论文写作要点.md').write_text('\n'.join(lines),encoding='utf-8')
    # Promote the reviewed active paper; all original paper evidence remains in immutable r14.
    for p in (OUT/'IEEE_experiments').rglob('*'):
        if p.is_file():
            dst=ROOT/'paper/phase1_experiments_ieee_20261020'/p.relative_to(OUT/'IEEE_experiments');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for srcfile,dststem in [('Fig22_F0_seven_views_restored','F0_seven_views'),('Fig08_source_cross_rx_restored','source_count')]:
        for ext,sub in [('svg','svg'),('png','preview')]:
            shutil.copy2(OUT/sub/(srcfile+'.'+ext),ROOT/'paper/phase1_experiments_ieee_20261020/images/practical_full'/(dststem+'.'+ext))
    shutil.copy2(ROOT/'tools/figures/build_practical_paper_kit.py',OUT/'reproduce/build_practical_paper_kit.py')
    shutil.copy2(Path(__file__),OUT/'reproduce'/Path(__file__).name)
    notice={'usage_status':'HISTORICAL_ONLY_WRONG_CHANNEL_FOR_CURRENT_COMPARISON','current_run':RUN,
      'reason':'Original external training augmentation differs from the user-selected practical ResidualChannel residual/post_sync/noEQ; use the new 2x2 Full matrix for current paper comparisons.',
      'scope':'Old external recognition performance and its paper projections only. Existing experiment statuses/events remain historical facts; G IQ caches, native PAMF, architecture timing and valid mechanism evidence are not invalidated.',
      'replacement':'automation_reports/CV-SincNet/'+RUN+'/full_results_20261024',
      'quarantine_manifest':'docs/project_governance/20261024_comparison_retirement/cleanup_manifest.json'}
    for run in OLD_RUNS:
        p=ROOT/'automation_reports/CV-SincNet'/run
        if p.exists():(p/'usage_notice.json').write_text(json.dumps(dict(old_run=run,**notice),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cleanup=json.loads((DOC/'cleanup_manifest.json').read_text(encoding='utf-8'))
    assert all(not (ROOT/r).exists() for r in cleanup['targets'])
    (DOC/'REPORT.md').write_text(f'''# 2026-10-24 对比实验整理与隔离

VERIFIED：旧外部训练增强不匹配当前 practical residual 要求；新论文统一采用 `{RUN}`。全部 627 行、11 契约、三类泛化及四组合保留。

新论文包：`exports/unified_figures/20261024-practical-residual-paper-r03`。22 张 Times New Roman 独立图，PDF/SVG 可编辑矢量、4800 dpi PNG/TIFF；25 页中文逐图说明；6 页 IEEE 实验部分。9 张有效 PAMF 历史机制/结构/时延图保留原口径。

原目录已回收 {cleanup['deleted_file_count']} 个文件，逻辑大小 {cleanup['logical_bytes']} 字节（约 13.98 GB，不等于实测磁盘释放量）。归档位于 `{cleanup['archive']}`，恢复映射 `{cleanup['restore_map']}`。删除前完整读回 {cleanup['file_count']} 个成员并核验 CRC，删除后逐目标确认原路径不存在。

旧混合图包同时包含正确的 PAMF 证据，因此完整归档，不能整包声称科学无效。保留模型、IQ、所有 Git 工作树、远端数据/进程、实验登记及恢复事件。旧外部注册记录增加 usage_notice.json，禁止作为当前比较；并未把历史 ANALYZED 改成虚假失败或删除必要追溯信息。新结果 r14 原包保留；其历史附件已有明确 historical_evidence 目录。

当前论文、当前方法和冻结输入的逐路径引用检查为 0；本地相关运行进程引用为 0。只对检查范围内的本地导出和旧外部聚合副本操作，没有执行远端删除。

恢复：解压恢复映射，按原相对路径从归档提取到新目录；核对成员大小和 CRC 后再恢复所需路径，禁止覆盖现有新成果。详见 cleanup_manifest.json 和隔离目录内说明。
''',encoding='utf-8')
    p=ROOT/'docs/project_governance/current.json';state=json.loads(p.read_text(encoding='utf-8-sig'))
    state['comparison_retirement']='20261024_comparison_retirement/cleanup_manifest.json';state['comparison_retirement_date']='2026-10-24'
    state['latest_paper_assets']='exports/unified_figures/20261024-practical-residual-paper-r03'
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'docs/project_governance/README.md';s=p.read_text(encoding='utf-8-sig')
    if '20261024_comparison_retirement/REPORT.md' not in s:s+='\n## 2026-10-24 对比结果整理\n\n新 practical residual 论文资料与旧混合导出的隔离、回收和恢复映射见[本次报告](20261024_comparison_retirement/REPORT.md)。原生模型、IQ、有效 PAMF 机制及远端记录保持。\n'
    p.write_text(s.rstrip()+'\n',encoding='utf-8')
    p=ROOT/'PROJECT_NAVIGATION.md';s=p.read_text(encoding='utf-8-sig')
    if '20261024_comparison_retirement/REPORT.md' not in s:s+='\n## 最新 practical residual 论文资料\n\n使用 `exports/unified_figures/20261024-practical-residual-paper-r03`。旧错误增强外部比较及混合包已隔离，原链接需结合[回收与恢复说明](docs/project_governance/20261024_comparison_retirement/REPORT.md)定位；当前完整冻结结果为 practical-residual r14。\n'
    p.write_text(s.rstrip()+'\n',encoding='utf-8')
    for rel in ('manuscript_results_en.md','figure_captions_en.md','论文写作要点.md','figure_manifest.json','verification.json'):
        shutil.copy2(OUT/rel,PUBLIC/rel)
    for d in ('pdf','svg','tables'):
        shutil.copytree(OUT/d,PUBLIC/d,dirs_exist_ok=True)
    shutil.copy2(DOC/'cleanup_manifest.json',OUT/'cleanup_manifest.json')
    shutil.copy2(DOC/'REPORT.md',OUT/'旧资料隔离说明.md')
    delivery={'status':'VERIFIED','current_run':RUN,'new_figure_count':22,'historical_PAMF_figures':9,
      'IEEE_pages':6,'Chinese_guide_pages':25,'figure_data_rows_checked':checked,'old_local_files_archived_then_deleted':cleanup['deleted_file_count'],
      'new_results_retrained_or_rescored':False,'current_paper_uses_only_correct_new_external_results':True}
    (OUT/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Create fresh packages. Full frozen new statistics are copied byte-for-byte from r14;
    # no obsolete historical_evidence subtree is copied into the new paper package.
    final=ROOT/'exports/phase1_external_dg/20261024-practical-residual-paper-complete-r15.zip'
    assert not final.exists(),str(final)
    with zipfile.ZipFile(final,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.relative_to(OUT).parts[0] not in ('png','tiff','paper_preview'):
                z.write(p,'paper_assets/'+p.relative_to(OUT).as_posix())
        prior=ROOT/'exports/phase1_external_dg/20261024-practical-residual-full2x2-complete-r14.zip'
        with zipfile.ZipFile(prior) as old:
            for x in old.infolist():
                if x.filename.startswith(('inputs/','results/','reproduce/','method_source/')):
                    z.writestr('frozen_statistics/'+x.filename,old.read(x))
            z.writestr('frozen_statistics/README.md',old.read('README.md').decode('utf-8').replace('historical_evidence 原样保留已验证 r12 科学输入/复算器和 r13 旧论文投影','原 r14 的 historical_evidence 在本包未复制；有效 PAMF 历史证据另列。原包原样保留 r12 科学输入/复算器和 r13 旧论文投影'))
        z.writestr('README.md','# practical residual paper package r15\n\nUse paper_assets for manuscript text, tables and figures. The 4800 dpi raster files are separate ZIPs. frozen_statistics contains only the verified new 627-row inputs/results/reproducer; obsolete historical external comparison packages are excluded.\n\nRecompute with Python >=3.10 from frozen_statistics/reproduce:\n\n    python -m experiments.practical_full_2x2.reproduce_full --input ../inputs --expected ../results --output ../../recomputed_new\n\nReplot from the extracted paper_assets:\n\n    python reproduce/build_practical_paper_kit.py --source source_data --historical historical_PAMF --output ../replotted --raster\n')
    packages=[final]
    for sub in ('png','tiff'):
        path=OUT.parent/(OUT.name+'-'+sub.upper()+'-4800dpi.zip');assert not path.exists()
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted((OUT/sub).glob('*')):z.write(p,sub+'/'+p.name)
        packages.append(path)
    results=[]
    for p in packages:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            assert not any(x.filename.startswith('historical_evidence/') for x in z.infolist())
            results.append(dict(path=str(p),bytes=p.stat().st_size,members=len(z.infolist()),full_CRC_readback=True))
    (PUBLIC/'package_verification.json').write_text(json.dumps({'status':'VERIFIED','packages':results,'statistics_bytes_reused_from_verified_r14':True,'old_panels_recomputed':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(delivery=delivery,packages=results),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
