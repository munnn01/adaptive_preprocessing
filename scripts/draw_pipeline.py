"""Draw the documented V34-R1 pipeline; runtime source is not altered."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
COLORS = dict(input=('#edf5e8','#66994c'), frozen=('#e9efff','#4268ad'),
              fixed=('#fff2d9','#b47d28'), trained=('#dbf5ec','#247d65'))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none'})
    fig, ax = plt.subplots(figsize=(17,9.3))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0,16.8), ylim=(0,9.3)); ax.axis('off')

    def text(x,y,value,size=11,weight='normal',color='#1d2a3c',ha='center'):
        return ax.text(x,y,value,ha=ha,va='center',fontsize=size,fontweight=weight,color=color,linespacing=1.5)

    def box(x,y,w,h,title,body,kind='fixed'):
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',
                                 linewidth=1.4,facecolor=fill,edgecolor=edge))
        text(x+w/2,y+h-.29,title,11.5,'bold')
        text(x+w/2,y+h/2-.17,body,10.3)

    def arrow(points,color='#354052',dashed=False,label=None):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.45,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,
                                    linewidth=1.45,color=color,linestyle='--' if dashed else '-'))
        if label:
            text(*label[:2],label[2],9.5,color=color)

    text(.35,8.85,'V34-R1  |  Adaptive preprocessing before standard codecs',21,'bold',ha='left')
    text(.35,8.4,'INFERENCE — frozen controller, fixed filters and frozen task weights',11.5,'bold',ha='left')
    # Sender/receiver boundaries.
    for x,w,title in [(.25,11.85,'SENDER: region-aware candidate search'),(12.28,4.2,'RECEIVER')]:
        ax.add_patch(FancyBboxPatch((x,4.33),w,3.8,boxstyle='round,pad=0.03,rounding_size=0.06',
                                   facecolor='#fafbfd',edgecolor='#c7d1df',linewidth=1))
        text(x+.15,7.89,title,10.5,'bold',ha='left')
    box(2.75,6.48,2.6,1.05,'Region maps','Detector + causal motion\nobject / context / background','input')
    box(6.1,6.48,2.65,1.05,'CNN + GRU','RGB + maps + 9 conditions\n13 logits, task risks, rates','frozen')
    box(.5,5.0,1.6,1.05,'Source','RGB clip','input')
    box(2.75,5.0,2.6,1.05,'Fixed filters','3 shortlisted edits + identity\ngeometry preserved','fixed')
    box(6.1,5.0,2.65,1.05,'Real codec trials','x264 / x265 encode\n+ sender decode','fixed')
    box(9.4,5.0,2.4,1.05,'Source guard','Consistency + rate\nidentity fallback','fixed')
    box(12.53,5.0,1.58,1.05,'Decoder','Independent\nH.264 / H.265','fixed')
    box(14.58,5.0,1.6,1.05,'Video Swin-T','Frozen AR\nprediction','frozen')
    arrow([(2.1,5.525),(2.68,5.525)])
    arrow([(5.42,5.525),(6.03,5.525)])
    arrow([(8.82,5.525),(9.33,5.525)])
    arrow([(11.87,5.525),(12.46,5.525)])
    text(12.17,6.38,'Selected packet\n+ common header',9.5)
    arrow([(14.18,5.525),(14.51,5.525)])
    arrow([(1.3,6.12),(1.3,7.0),(2.68,7.0)])
    arrow([(5.42,7.0),(6.03,7.0)])
    arrow([(1.3,7.0),(1.3,7.67),(7.42,7.67),(7.42,7.6)])
    text(4.4,7.78,'RGB',9.5)
    arrow([(4.05,6.41),(4.05,6.12)])
    arrow([(6.1,6.69),(5.7,6.69),(5.7,6.24),(4.8,6.24),(4.8,6.12)])
    text(7.42,6.26,'codec / QP / budget\ntask mask / domain',8.5)
    arrow([(1.3,4.93),(1.3,4.58),(10.6,4.58),(10.6,4.93)],'#677b92',True,
          (6.0,4.75,'Clean-source reference: sender only; no source RGB or maps sent'))
    text(.4,3.98,'Anchor: unchanged RGB → same codec / QP / GOP → same decoder → same frozen Video Swin-T',11,ha='left')

    text(.35,3.32,'OFFLINE TRAINING — real codec measurements become fixed supervision',11.5,'bold',ha='left')
    box(.5,1.83,2.1,1.08,'TRAIN sources','RGB + region maps\nGT where available','input')
    box(3.18,1.83,2.65,1.08,'Measure 13 profiles','Real x264 / x265\nFrozen AR / OT task heads','fixed')
    box(6.43,1.83,2.55,1.08,'Cached targets','Action / task risk / rate\nSTOP GRADIENT','frozen')
    box(9.58,1.83,2.65,1.08,'Train CNN + GRU θ','Action CE + risk Huber\n+ relative-rate MSE','trained')
    box(12.95,1.83,3.15,1.08,'Frozen LAST checkpoints','303401 / 303402 / 303403\n+ domain CAL residual metadata','frozen')
    arrow([(2.67,2.37),(3.11,2.37)])
    arrow([(5.9,2.37),(6.36,2.37)])
    arrow([(9.05,2.37),(9.51,2.37)],'#677b92',True)
    arrow([(12.3,2.37),(12.88,2.37)])
    arrow([(1.55,1.76),(1.55,1.22),(10.4,1.22),(10.4,1.76)])
    text(5.45,1.42,'TRAIN RGB + maps + conditions',9.5)
    arrow([(11.55,1.76),(11.55,.72),(10.95,.72),(10.95,1.76)],'#c54940',True)
    text(12.88,.73,'Backpropagation: θ only',10.5,'bold','#c54940')
    text(.5,.43,'No STE or virtual-codec gradient in V34. Independent evaluation: AR only; OT/OD GT is not available in this validation set.',10,ha='left')
    destination = ROOT/'docs/figures'
    destination.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination/'v34_pipeline.png',dpi=150,bbox_inches='tight',pad_inches=.16)
    fig.savefig(destination/'v34_pipeline.svg',bbox_inches='tight',pad_inches=.16,
                metadata={'Title':'V34-R1 adaptive preprocessing pipeline',
                          'Description':'Sender-only region-aware filter selection using a frozen CNN GRU controller and real codec trials; standard decoder and frozen Video Swin-T at receiver. Offline fixed measurements supervise only controller parameters.'})
    svg = destination/'v34_pipeline.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode())
    plt.close(fig)


if __name__ == '__main__':
    main()
