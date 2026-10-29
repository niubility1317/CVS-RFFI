"""Draw editable scientific schematics from the fixed PAMF/DURA method definitions."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Polygon
from matplotlib.path import Path as MPath

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
C={'input':'#edf2f5','pamf':'#f6e7ec','dura':'#fbf0d9','train':'#e9f2ef','white':'#ffffff','edge':'#516878','text':'#263943','red':'#b33350','orange':'#a86a08'}

def canvas(height=4.1):
    fig,ax=plt.subplots(figsize=(7.1,height));ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off');fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985);return fig,ax

def box(ax,x,y,w,h,label,kind='white',font=7.3,bold=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=1.1',facecolor=C[kind],edgecolor=C['edge'],linewidth=.8,zorder=3)
    ax.add_patch(patch);txt=ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=font,color=C['text'],weight='bold' if bold else 'normal',linespacing=1.23,zorder=4)
    return patch,txt

def line(ax,points,style='-',color=None,arrow=True):
    p=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>' if arrow else '-',mutation_scale=8,linewidth=.85,linestyle=style,color=color or C['edge'],zorder=2))

def text(ax,x,y,s,sz=7.1,ha='left',color=None,weight='normal'):
    return ax.text(x,y,s,fontsize=sz,ha=ha,va='center',color=color or C['text'],weight=weight,zorder=4)

def save(fig,out,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

def pamf_a(out):
    fig,ax=canvas(4.4)
    text(ax,1,96,'Inference architecture: complementary waveform representations',9,weight='bold')
    box(ax,1,40,13,14,'Received IQ\nI / Q window','input')
    labels=[(76,'Temporal encoder\nSinc + CNN'),(57,'Mirrored spectrum\npower and asymmetry'),(38,'Nonlinear memory\nPA-inspired branch'),(19,'Reference response\nknown preamble')]
    for y,label in labels:box(ax,23,y,20,12,label,'pamf')
    line(ax,[(14,47),(19,47)],arrow=False);line(ax,[(19,25),(19,82)],arrow=False)
    for y,_ in labels:line(ax,[(19,y+6),(23,y+6)])
    box(ax,49,75,20,13,'Three SIG-TIME views\nproject • reorder • encode','pamf')
    box(ax,49,54,20,13,'Time–frequency\nfeature fusion','pamf')
    box(ax,75,38,12,16,'Residual\nfusion','pamf',bold=True)
    box(ax,90,38,9,16,'Cosine\nhead','pamf')
    line(ax,[(43,82),(49,82)])
    line(ax,[(43,79),(46,79),(46,63),(49,63)])
    line(ax,[(43,63),(49,59)])
    line(ax,[(69,81),(72,81),(72,51),(75,51)])
    line(ax,[(69,60),(71,60),(71,48),(75,48)])
    line(ax,[(43,44),(75,44)])
    line(ax,[(43,25),(72,25),(72,41),(75,41)])
    line(ax,[(87,46),(90,46)])
    text(ax,49,8,'The three views share one intermediate sequence; raw IQ is not permuted.',7.3,ha='center')
    save(fig,out,'PAMF_A_parallel_branches')

def pamf_b(out):
    fig,ax=canvas(4.0)
    text(ax,1,96,'Three-view SIG-TIME: local feature-path encoding',9,weight='bold')
    box(ax,1,53,17,19,'Temporal feature\nintermediate t2','pamf')
    for i,y in enumerate([75,53,31],1):
        box(ax,27,y,16,14,f'View {i}\nlearned projection','pamf')
        line(ax,[(18,62),(22,62),(22,y+7),(27,y+7)])
    box(ax,51,48,19,27,'Local windows\nfixed permutation\n+ progress coordinate','pamf')
    for y in [82,60,38]:line(ax,[(43,y),(47,y),(47,61),(51,61)])
    box(ax,75,48,23,27,'Increments + areas\nconcatenate views\nproject to feature','pamf')
    line(ax,[(70,61),(75,61)])
    box(ax,8,7,38,14,'Temporal + spectral • PA • reference features','pamf')
    box(ax,57,7,23,14,'Residual fusion','pamf',bold=True)
    box(ax,86,7,13,14,'Cosine head','pamf')
    line(ax,[(46,14),(57,14)]);line(ax,[(80,14),(86,14)])
    line(ax,[(87,48),(87,30),(68,30),(68,21)])
    text(ax,33,25,'Permutation acts on features, not input IQ.',7.1,ha='center')
    save(fig,out,'PAMF_B_signature_focus')

def pamf_c(out):
    fig,ax=canvas(4.0)
    text(ax,1,96,'One identifier, two roles: training and frozen inference',9,weight='bold')
    text(ax,2,82,'TRAINING ONLY',7.6,weight='bold')
    box(ax,2,59,18,18,'Labeled source IQ\ntrue identities','input')
    box(ax,2,28,18,18,'Unlabeled source IQ\nTX identities hidden','input')
    box(ax,26,59,21,18,'PAMF forward\nlabeled objective','train')
    box(ax,26,28,21,18,'Current student\npseudo-label filter','train')
    box(ax,52,28,21,18,'Cross-RX pairing\nmix / contrast / extrapolate','train',font=7)
    box(ax,79,42,19,22,'Labeled-gradient\nanchoring and\nparameter update','train')
    line(ax,[(20,68),(26,68)]);line(ax,[(20,37),(26,37)]);line(ax,[(47,37),(52,37)])
    line(ax,[(47,68),(75,68),(75,57),(79,57)]);line(ax,[(73,37),(76,37),(76,49),(79,49)])
    text(ax,2,22,'INFERENCE',7.6,weight='bold')
    box(ax,2,4,18,16,'Received IQ\nunseen receiver','input')
    box(ax,26,4,24,16,'Frozen PAMF\nstructured branches','pamf')
    box(ax,60,4,19,16,'Original cosine\nclassifier','pamf')
    box(ax,86,4,12,16,'TX identity','input')
    line(ax,[(20,12),(26,12)]);line(ax,[(50,12),(60,12)]);line(ax,[(79,12),(86,12)])
    line(ax,[(88,42),(88,26),(38,26),(38,20)],style='--')
    text(ax,52,23,'freeze',6.5,ha='center')
    save(fig,out,'PAMF_C_training_inference')

def dura_a(out):
    fig,ax=canvas(3.8)
    text(ax,1,96,'DURA: coarse estimate and two data-consistent stages',9,weight='bold')
    box(ax,1,41,13,17,'Received IQ\ny','input')
    box(ax,21,67,20,17,'Coarse estimator\nwaveform c','dura')
    box(ax,21,25,20,20,'Action estimator\noperator A + variance v','dura')
    line(ax,[(14,50),(18,50),(18,75),(21,75)]);line(ax,[(14,50),(18,50),(18,35),(21,35)])
    line(ax,[(31,67),(31,45)])
    box(ax,51,52,20,24,'Stage 1\nprior + data consistency\n2 CG iterations','dura')
    box(ax,75,52,20,24,'Stage 2\nprior + data consistency\n2 CG iterations','dura')
    line(ax,[(41,75),(51,68)]);line(ax,[(71,64),(75,64)])
    line(ax,[(41,35),(46,35),(46,58),(51,58)])
    line(ax,[(41,35),(73,35),(73,56),(75,56)])
    box(ax,75,8,20,15,'Restored IQ\nfor frozen classifier','input')
    line(ax,[(85,52),(85,23)])
    text(ax,39,12,'Both stages share y, c, A, and v.',7.1,ha='center')
    save(fig,out,'DURA_A_two_stage')

def dura_b(out):
    fig,ax=canvas(3.7)
    text(ax,1,96,'One restoration stage: learned prior and physical consistency',9,weight='bold')
    box(ax,2,55,18,20,'Current estimate\nprevious stage','dura')
    box(ax,27,55,20,20,'Back-projected\nobservation residual','dura')
    box(ax,54,55,19,20,'Learned prior\nwaveform proposal','dura')
    box(ax,80,55,18,20,'Data-consistency\nCG update','dura')
    for x in [20,47,73]:line(ax,[(x,65),(x+7,65)])
    box(ax,3,12,25,20,'Current observation y\ncoarse estimate c','input')
    box(ax,39,12,26,20,'Estimated action A\nwindow variance v','dura')
    box(ax,76,12,21,20,'Output estimate\nto next stage','dura')
    line(ax,[(15,32),(15,45),(37,45),(37,55)]);line(ax,[(52,32),(52,45),(88,45),(88,55)])
    line(ax,[(39,22),(33,22),(33,55)])
    line(ax,[(89,55),(89,32)])
    text(ax,50,4,'Prior also receives y, c, the current estimate, and v; consistency is soft.',6.8,ha='center')
    save(fig,out,'DURA_B_stage_detail')

def system_a(out):
    fig,ax=canvas(4.4)
    text(ax,1,96,'Source development and spaceborne inference',9,weight='bold')
    text(ax,2,84,'SOURCE-ONLY DEVELOPMENT',7.4,weight='bold')
    box(ax,2,62,22,17,'Source IQ\nlabeled + unlabeled','input')
    box(ax,32,73,26,16,'Train PAMF\nTri-SIG-TIME SSL','pamf')
    box(ax,32,48,26,16,'Train DURA\nL or L+U source IQ pairs\n(no TX labels)','dura',font=6.9)
    box(ax,68,73,27,16,'Frozen PAMF\noriginal classifier','pamf')
    box(ax,68,48,27,16,'Frozen DURA\noptional front end','dura')
    line(ax,[(24,70),(28,70),(28,81),(32,81)]);line(ax,[(24,70),(28,70),(28,56),(32,56)])
    line(ax,[(58,81),(68,81)],style='--');line(ax,[(58,56),(68,56)],style='--')
    text(ax,2,35,'PASSIVE INFERENCE',7.4,weight='bold')
    box(ax,2,8,22,17,'Unseen receiver\nterrestrial emitter IQ','input')
    box(ax,34,8,25,17,'Optional DURA\nIQ restoration','dura')
    box(ax,68,8,17,17,'Frozen PAMF','pamf')
    box(ax,90,8,9,17,'TX ID','input')
    line(ax,[(24,16),(34,16)]);line(ax,[(59,16),(68,16)]);line(ax,[(85,16),(90,16)])
    line(ax,[(24,19),(28,19),(28,28),(64,28),(64,19),(68,19)])
    line(ax,[(95,81),(98,81),(98,32),(76,32),(76,25)],style='--')
    line(ax,[(81,48),(81,40),(47,40),(47,25)],style='--')
    text(ax,50,30,'direct path',6.7,ha='center');text(ax,97,43,'Frozen',6.3,ha='right')
    save(fig,out,'System_A_training_and_inference')

def system_b(out):
    fig,ax=canvas(3.2)
    text(ax,1,95,'Spaceborne RF fingerprinting: two inference configurations',9,weight='bold')
    box(ax,2,48,18,20,'Terrestrial emitter\nreceived IQ','input')
    box(ax,29,68,24,20,'Frozen PAMF\ndirect identification','pamf')
    box(ax,29,14,24,20,'Optional DURA\nIQ restoration','dura')
    box(ax,62,14,20,20,'Frozen PAMF\nidentification','pamf')
    box(ax,88,41,11,24,'Emitter\nidentity','input')
    line(ax,[(20,58),(24,58),(24,78),(29,78)]);line(ax,[(20,58),(24,58),(24,24),(29,24)])
    line(ax,[(53,24),(62,24)]);line(ax,[(53,78),(85,78),(85,57),(88,57)])
    line(ax,[(82,24),(85,24),(85,49),(88,49)])
    text(ax,50,7,'Target IQ is used for inference; no target-domain updating.',7.1,ha='center')
    save(fig,out,'System_B_inference_options')

def system_c(out):
    fig,ax=canvas(3.8)
    text(ax,1,95,'Concept: spaceborne passive RF fingerprinting',9,weight='bold')
    # A simple transmitter and satellite icon indicate the intended deployment.
    line(ax,[(10,18),(10,34)],arrow=False);line(ax,[(6,18),(14,18)],arrow=False)
    ax.add_patch(Polygon([[6,33],[14,33],[10,41]],closed=True,facecolor=C['input'],edgecolor=C['edge'],lw=.8))
    text(ax,10,9,'Terrestrial\nemitter',7.0,ha='center')
    ax.add_patch(Rectangle((24,67),7,12,facecolor=C['input'],edgecolor=C['edge'],lw=.8))
    ax.add_patch(Rectangle((17,69),6,8,facecolor='#d9e7ee',edgecolor=C['edge'],lw=.8))
    ax.add_patch(Rectangle((32,69),6,8,facecolor='#d9e7ee',edgecolor=C['edge'],lw=.8))
    line(ax,[(10,40),(24,66)],style='--');text(ax,7,55,'Space–ground\npropagation',6.6)
    text(ax,27.5,85,'Satellite receiver',7.1,ha='center')
    box(ax,43,55,16,20,'Received\nIQ window','input')
    box(ax,64,55,16,20,'Optional\nDURA','dura')
    box(ax,84,55,15,20,'Frozen\nPAMF','pamf')
    box(ax,84,15,15,18,'Emitter\nidentity','input')
    line(ax,[(38,73),(43,65)]);line(ax,[(59,65),(64,65)]);line(ax,[(80,65),(84,65)])
    line(ax,[(51,75),(51,83),(91.5,83),(91.5,75)])
    text(ax,72,88,'direct path',6.6,ha='center');line(ax,[(91.5,55),(91.5,33)])
    text(ax,48,8,'Deployment concept; evaluation uses terrestrial IQ with simulated channel stress.',6.8,ha='center')
    save(fig,out,'System_C_spaceborne_context')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for fn in [pamf_a,pamf_b,pamf_c,dura_a,dura_b,system_a,system_b,system_c]:fn(args.output)
    manifest={'backend':'Python / Matplotlib','formats':['PDF','SVG','PNG'],'figures':[
      {'id':'PAMF_A_parallel_branches','role':'Whole inference architecture; candidate for the main paper'},
      {'id':'PAMF_B_signature_focus','role':'Three-view local path encoding and its connection to fusion'},
      {'id':'PAMF_C_training_inference','role':'Distinguish source-only SSL operations from the deployed network'},
      {'id':'DURA_A_two_stage','role':'Coarse estimation, action estimation, and two unrolled stages'},
      {'id':'DURA_B_stage_detail','role':'Inside one prior-plus-consistency stage'},
      {'id':'System_A_training_and_inference','role':'Source development and optional restoration at inference'},
      {'id':'System_B_inference_options','role':'Direct and restored inference configurations'},
      {'id':'System_C_spaceborne_context','role':'Conceptual satellite/ground sensing and inference paths'}]}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
