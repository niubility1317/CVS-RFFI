"""Compose selected r3 vector artwork without redrawing or recoloring experiments."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import pymupdf as fitz

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    for key in ['catalog','results','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--plot-python',default=sys.executable)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'data').mkdir(exist_ok=True)
    # Only the methodological diagram is authored; experimental artwork is reused.
    if not (a.output/'fig01.pdf').exists():
        subprocess.run([a.plot_python,str(Path(__file__).with_name('render_pamf_paper_figures_20261029.py')),'--catalog',str(a.catalog),'--results',str(a.results),'--output',str(a.output),'--only-figure','1'],check=True)
    # Source, normalized clip, heading, source-data panel filter.
    def tile(stem,clip=(0,0,1,1),heading='',panel=None):return dict(stem=stem,clip=clip,heading=heading,panel=panel)
    plans={
      2:(1,[tile('Fig001A_network_cross_rx',(0,0,1,.5),'Cross-receiver, clean','a'),tile('Fig001_network_accuracy_f1',(0,0,1,.5),'Receiver + date holdout, clean','a')]),
      3:(1,[tile('Fig022_learning_augmented_cross_rx',(.5,0,1,1),'Cross-receiver: augmented learning regimes','b'),tile('Fig031_learning_augmented_cross_both',(.5,0,1,1),'Receiver + date holdout: augmented learning regimes','b')]),
      4:(1,[tile('Fig025_seven_scenes_cross_rx',heading='Cross-receiver'),tile('Fig034_seven_scenes_cross_both',heading='Receiver + date holdout')]),
      5:(2,[tile('Fig023_receiver_robustness_cross_rx',(.5,0,1,1),'Cross-receiver: receiver splits','b'),tile('Fig032_receiver_robustness_cross_both',(.5,0,1,1),'Receiver + date holdout: receiver splits','b'),tile('Fig024_label_robustness_cross_rx',(.5,0,1,1),'Cross-receiver: label fraction','b'),tile('Fig033_label_robustness_cross_both',(.5,0,1,1),'Receiver + date holdout: label fraction','b')]),
      6:(1,[tile('Fig027_source_count_practical_cross_rx',heading='Cross-receiver: six-scene mean'),tile('Fig036_source_count_practical_cross_both',heading='Receiver + date holdout: six-scene mean')]),
      7:(1,[tile('Fig029B_Fig038_DURA_gains')]),
      8:(1,[tile('Fig039_branch_ce_paired',heading='CE: receiver + date holdout'),tile('Fig043_branch_ssl_paired',heading='Complete SSL: receiver + date holdout')])}
    manifest=[{'paper_figure':1,'sources':['Current-method training/inference diagram'],'data':None}]
    checks=[]
    for number,(cols,tiles) in plans.items():
        width=720;gap=14;cell_width=(width-gap*(cols-1))/cols
        docs=[];clips=[];heights=[];color_sets=[]
        for t in tiles:
            doc=fitz.open(a.catalog/'pdf'/(t['stem']+'.pdf'));page=doc[0];w,h=page.rect.width,page.rect.height
            clip=fitz.Rect(t['clip'][0]*w,t['clip'][1]*h,t['clip'][2]*w,t['clip'][3]*h)
            docs.append(doc);clips.append(clip)
            heights.append(cell_width*clip.height/clip.width+(23 if t['heading'] else 0))
            colors=set()
            for drawing in page.get_drawings():
                for field in ['color','fill']:
                    c=drawing.get(field)
                    if c:colors.add(tuple(round(v,6) for v in c))
            color_sets.append(colors)
        row_heights=[max(heights[i:i+cols]) for i in range(0,len(tiles),cols)]
        height=sum(row_heights)+gap*(len(row_heights)-1)
        result=fitz.open();page=result.new_page(width=width,height=height);y=0;data=[]
        for i,t in enumerate(tiles):
            row,col=divmod(i,cols)
            if col==0 and row:y+=row_heights[row-1]+gap
            x=col*(cell_width+gap);top=y
            if t['heading']:
                page.insert_text((x+3,top+14),t['heading'],fontsize=10,color=(0,0,0));top+=23
            rect=fitz.Rect(x,top,x+cell_width,top+cell_width*clips[i].height/clips[i].width)
            page.show_pdf_page(rect,docs[i],0,clip=clips[i],keep_proportion=True)
            rows=read(a.catalog/'source_data'/(t['stem']+'_plot.csv'))
            if t['panel'] is not None:rows=[r for r in rows if r['panel']==t['panel']]
            data.extend(dict(source_figure=t['stem'],paper_tile=i+1,**r) for r in rows)
        path=a.output/f'fig{number:02}.pdf';result.save(path,garbage=4,deflate=True)
        result[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(a.output/f'fig{number:02}.png')
        if number==2:
            by_tile=[[r for r in data if r['paper_tile']==i] for i in [1,2]]
            assert all(len(rows)==15 and len({r['category'] for r in rows})==15 for rows in by_tile)
            assert all(int(r['seed_n'])==8 for r in data)
            assert {r['test'] for r in by_tile[0]}=={'cross_rx'}
            assert {int(r['n']) for r in by_tile[0]}=={126000}
            assert {r['test'] for r in by_tile[1]}=={'cross_both'}
            count_rows=[r for r in read(a.results/'15_architecture_E200_summary.csv') if r['test']=='cross_both']
            assert len(count_rows)==15 and {r['network'] for r in count_rows}=={r['category'] for r in by_tile[1]}
            assert {int(r['query_count']) for r in count_rows}=={42000}
        # Embedded PDF drawing colors remain source RGB values. Only new headings are black.
        allowed=set().union(*color_sets)|{(0.,0.,0.),(1.,1.,1.)}
        actual=set()
        for drawing in result[0].get_drawings():
            for field in ['color','fill']:
                c=drawing.get(field)
                if c:actual.add(tuple(round(v,6) for v in c))
        assert actual<=allowed,(number,'Unexpected color transformation',actual-allowed)
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (a.output/'data'/f'fig{number:02}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
        manifest.append({'paper_figure':number,'sources':[t['stem'] for t in tiles],'data':f'data/fig{number:02}.csv','source_artwork':'selected_r3/pdf','composition':tiles,'redrawn':False,'recolored':False})
        checks.append({'figure':number,'original_artwork':True,'colors_preserved':True,'tiles':len(tiles),'rows':len(data)})
        result.close()
        for doc in docs:doc.close()
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'artwork_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))

if __name__=='__main__':main()
