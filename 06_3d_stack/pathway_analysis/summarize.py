"""Summarize Study 06 class-marker and GO-enrichment preservation."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import textwrap
os.environ.setdefault('MPLCONFIGDIR','/tmp/study06-go-matplotlib')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
GAPS=(3,5,10)
COLORS={3:'#0072B2',5:'#E69F00',10:'#009E73'}
FDR=.05


def compare(root):
    inventory=pd.read_csv(root/'ranking_inventory.csv')
    rows,terms,selected=[],[],[]
    for gap in GAPS:
        scores={method:pd.read_csv(root/f'gap{gap}/{method}_rankings.csv.gz',index_col='gene') for method in ('observed','feast')}
        assert scores['observed'].index.equals(scores['feast'].index)
        for label in scores['observed'].columns:
            meta=inventory[(inventory.gap==gap)&(inventory.label==label)]
            real_meta=meta[meta.method=='observed'].iloc[0]
            gen_meta=meta[meta.method=='feast'].iloc[0]
            real=pd.read_csv(real_meta.result_file).set_index('term').sort_index()
            gen=pd.read_csv(gen_meta.result_file).set_index('term').sort_index()
            assert real.index.equals(gen.index)
            rp=(real.nes>0)&(real.gsea_fdr_q<FDR);gp=(gen.nes>0)&(gen.gsea_fdr_q<FDR)
            common=int((rp&gp).sum());union=int((rp|gp).sum())
            rows.append(dict(gap=gap,label=label,n_label_cells=int(real_meta.n_label_cells),
                gene_rank_spearman=float(spearmanr(scores['observed'][label],scores['feast'][label]).statistic),
                nes_spearman=float(spearmanr(real.nes,gen.nes).statistic),
                nes_mae=float(np.mean(np.abs(real.nes-gen.nes))),
                pathway_direction_agreement=float(np.mean(np.sign(real.nes)==np.sign(gen.nes))),
                observed_positive_terms=int(rp.sum()),feast_positive_terms=int(gp.sum()),
                recovered_positive_terms=common,positive_recall=common/int(rp.sum()) if rp.any() else np.nan,
                positive_precision=common/int(gp.sum()) if gp.any() else np.nan,
                positive_jaccard=common/union if union else np.nan))
            frame=pd.DataFrame({'term':real.index,'observed_nes':real.nes.to_numpy(),'feast_nes':gen.nes.to_numpy(),
                'observed_q':real.gsea_fdr_q.to_numpy(),'feast_q':gen.gsea_fdr_q.to_numpy(),
                'observed_leading_edge':real.leading_edge_genes.to_numpy(),'feast_leading_edge':gen.leading_edge_genes.to_numpy(),
                'gap':gap,'label':label})
            terms.append(frame)
            if gap==3:
                candidates=frame[(frame.observed_nes>0)&(frame.observed_q<FDR)].sort_values(
                    ['observed_q','observed_nes','term'],ascending=[True,False,True])
                if len(candidates):selected.append(candidates.iloc[0].to_dict())
    summary=pd.DataFrame(rows);long=pd.concat(terms,ignore_index=True)
    summary.to_csv(root/'preservation_metrics.csv',index=False)
    long.to_csv(root/'term_comparison.csv.gz',index=False)
    pd.DataFrame(selected).to_csv(root/'example_term_selection.csv',index=False)
    return summary,long,pd.DataFrame(selected)


def style():
    for f in ('arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf'):
        font_manager.fontManager.addfont(Path('/maiziezhou_lab2/yiru/miniconda3/fonts')/f)
    mpl.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':12,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})


def export(fig,out,stem,bundle):
    bundle.savefig(fig,bbox_inches='tight')
    for extension in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else {}
        fig.savefig(out/f'{stem}.{extension}',dpi=600,bbox_inches='tight',metadata=metadata)
    plt.close(fig)


def figures(summary,long,selected,out):
    out.mkdir(parents=True,exist_ok=True);style()
    with PdfPages(out/'study06_biological_preservation.pdf') as bundle:
        fig,axes=plt.subplots(1,3,figsize=(12.5,10.4),sharey=True)
        labels=sorted(summary.label.unique())
        specs=[('gene_rank_spearman','Class-marker ranking\nSpearman correlation'),
               ('nes_spearman','GO enrichment profile\nSpearman correlation'),
               ('positive_recall','Recovery of observed positive terms\nRecall')]
        for axis,(metric,title) in zip(axes,specs):
            matrix=summary.pivot(index='label',columns='gap',values=metric).reindex(index=labels,columns=GAPS)
            cmap=mpl.colormaps['viridis'].copy();cmap.set_bad('#DDDDDD')
            lower=-1 if metric!='positive_recall' else 0
            im=axis.imshow(matrix.to_numpy(),aspect='auto',cmap=cmap,vmin=lower,vmax=1)
            axis.set_xticks(range(3),['3','5','10']);axis.set_xlabel('Reference gap')
            axis.set_yticks(range(len(labels)),labels,fontsize=8.5)
            axis.set_title(title,fontweight='bold',pad=12)
            axis.tick_params(length=0)
            for y in range(len(labels)):
                for x in range(3):
                    value=matrix.iloc[y,x]
                    if np.isfinite(value):axis.text(x,y,f'{value:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if value<.5 else '#222222')
            fig.colorbar(im,ax=axis,orientation='horizontal',fraction=.03,pad=.065)
        fig.suptitle('Preservation of class-specific gene and GO signatures',fontsize=17,fontweight='bold',y=.985)
        fig.subplots_adjust(left=.215,right=.98,top=.89,bottom=.10,wspace=.20)
        fig.text(.5,.015,'Matched held-out cells only. Class labels condition generation; these are descriptive preservation scores.',ha='center',fontsize=10)
        export(fig,out,'class_signature_preservation',bundle)
        for page_start in range(0, len(selected), 12):
            fig,axes=plt.subplots(1,3,figsize=(15,7.5),sharey=True)
            ordered=selected.sort_values('label').iloc[page_start:page_start+12]
            labels=[r.label+'\n'+'\n'.join(textwrap.wrap(r.term.removeprefix('GOBP_').replace('_',' ').lower(),40)) for r in ordered.itertuples()]
            for axis,gap in zip(axes,GAPS):
                for y,entry in enumerate(ordered.itertuples()):
                    record=long[(long.gap==gap)&(long.label==entry.label)&(long.term==entry.term)].iloc[0]
                    axis.plot([record.observed_nes,record.feast_nes],[y,y],color='#B8BFC6',lw=1)
                    axis.scatter(record.observed_nes,y,c='#454D56',s=22,marker='s',label='Observed' if y==0 else None)
                    axis.scatter(record.feast_nes,y,c=COLORS[gap],s=28,label='FEAST' if y==0 else None)
                axis.axvline(0,color='#C9CED4',lw=.8)
                axis.set_title(f'Gap {gap}',color=COLORS[gap],fontweight='bold')
                axis.set_xlabel('Normalized enrichment score')
                axis.set_yticks(range(len(labels)),labels,fontsize=7.5)
                axis.grid(axis='x',color='#E5E8EB',lw=.6)
                axis.legend(loc='lower right')
            axes[0].invert_yaxis()
            fig.suptitle('Examples of observed class-associated GO programs',fontsize=17,fontweight='bold',y=.985)
            fig.subplots_adjust(left=.32,right=.99,top=.94,bottom=.065,wspace=.15)
            fig.text(.5,.015,'One term per class, selected only from positive observed gap-3 enrichment (q < 0.05). No selection by FEAST performance.',ha='center',fontsize=9)
            export(fig,out,f'class_go_examples_{page_start//12+1}',bundle)
    median=summary.groupby('gap')[['gene_rank_spearman','nes_spearman','positive_recall','positive_precision']].median()
    (out/'REPORT.md').write_text('# Study 06 biological-signature preservation\n\n```text\n'
        +median.to_string(float_format=lambda value:f'{value:.3f}')+'\n```\n\n'
        'These are class-versus-rest rankings and within-panel GO enrichment profiles, using matched held-out cells. '
        'All 34 broad class labels were supplied to the simulator. Agreement therefore supports conditional biological '
        'consistency, not independent cell-type discovery. Cells, serial slices, GO terms and generated outputs are '
        'not biological donor replicates. GSEA q-values describe gene-set enrichment within each ranking; they are '
        'not donor-level differential-expression evidence. The measured 1,122-gene panel is the universe, and term '
        'coverage is recorded in gene_set_coverage.csv. No new cell/gene filtering or DEG p-value selection is used.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    parser.add_argument('--output-dir',type=Path,default=HERE/'figures')
    args=parser.parse_args()
    status=json.loads((args.results/'completion.json').read_text())
    if status['status']!='complete':raise ValueError('complete GO results are required')
    summary,long,selected=compare(args.results);figures(summary,long,selected,args.output_dir)
    print('Completed Study 06 marker/GO comparison tables and figures.',flush=True)

if __name__=='__main__':main()
