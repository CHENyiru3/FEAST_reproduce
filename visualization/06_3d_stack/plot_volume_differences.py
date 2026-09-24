"""Whole Study 06 volumes, matched spatial deltas and marker-program profiles."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/study06-volume-matplotlib')
import h5py
from anndata.experimental import read_elem
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

import plot_3d_gene_distribution as volume
from plot_results import DEFAULT_INPUT, COLORS, configure_style, save, style_axis

GENES = volume.GENES
GAPS = (3, 5, 10)
CAP = volume.SAMPLE_CAP_PER_SLICE
SEED = 2026
DELTA_CMAP = LinearSegmentedColormap.from_list('study06_delta',
    ['#2166AC', '#67A9CF', '#E0E0E0', '#EF8A62', '#B2182B'], N=256)


def read_slice(path):
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        coordinates = np.asarray(read_elem(f['obsm/spatial_3d']), dtype=float)
        indices = var.index.get_indexer(GENES)
        if np.any(indices < 0):
            raise ValueError(f'{path}: missing predefined marker gene')
        counts = read_elem(f['layers/counts'])[:, indices]
        values = counts.toarray() if hasattr(counts, 'toarray') else np.asarray(counts)
    return obs, coordinates, values.astype(float, copy=False)


def collect(root, plan, output):
    points, errors, profiles, deltas = [], [], [], []
    donors = set()
    targets = {int(d['spec']['gap']): (name, set(map(int, d['reference_pool'])))
               for name, d in plan['densities'].items()}
    for number, slice_id in enumerate(map(int, plan['available_slices']), start=1):
        name = f'Zhuang-ABCA-1.{slice_id:03d}'
        obs, xyz, real = read_slice(Path(plan['data_dir']) / f'{name}.h5ad')
        if 'donor_id' in obs:
            donors.update(obs['donor_id'].astype(str).unique())
        selected = volume.sampled_indices(len(obs), SEED + slice_id)
        labels, codes = np.unique(obs['class'].astype(str).to_numpy(), return_inverse=True)
        group_n = np.bincount(codes)
        for gap in (0, *GAPS):
            is_reference = gap == 0 or slice_id in targets[gap][1]
            if is_reference:
                values = real
            else:
                generated_obs, generated_xyz, values = read_slice(root / 'targets' / targets[gap][0] / name / 'generated.h5ad')
                if not obs.index.equals(generated_obs.index) or not np.array_equal(xyz, generated_xyz):
                    raise ValueError(f'{name}, gap {gap}: observation identity or native geometry differs')
            delta = None if is_reference else np.log1p(values) - np.log1p(real)
            if delta is not None:
                deltas.append(delta)
                for index, gene in enumerate(GENES):
                    column = delta[:, index]
                    errors.append({'gap': gap, 'target_slice': slice_id, 'z': xyz[0, 2],
                        'gene': gene, 'n_spots': len(obs), 'mean_delta': column.mean(),
                        'mean_absolute_delta': np.abs(column).mean(),
                        'rmse_delta': np.sqrt(np.mean(column ** 2)),
                        'fraction_over': np.mean(column > 0), 'fraction_under': np.mean(column < 0)})
            frame = pd.DataFrame({'gap': gap, 'target_slice': slice_id,
                'spot_index': selected, 'x': xyz[selected, 0], 'y': xyz[selected, 1],
                'z': xyz[selected, 2], 'source_kind': 'real' if is_reference else 'simulated'})
            for index, gene in enumerate(GENES):
                frame[gene] = values[selected, index]
                frame[f'{gene}_delta'] = np.nan if delta is None else delta[selected, index]
                sums = np.bincount(codes, weights=values[:, index])
                for group, count, total in zip(labels, group_n, sums):
                    profiles.append({'gap': gap, 'target_slice': slice_id, 'z': xyz[0, 2],
                        'source_kind': 'real' if is_reference else 'simulated',
                        'class': group, 'gene': gene, 'n_spots': int(count), 'count_sum': float(total)})
            points.append(frame)
        if number % 12 == 0:
            print(f'Collected {number}/144 matched slice positions.', flush=True)
    limits = {}
    for index, gene in enumerate(GENES):
        absolute = np.abs(np.concatenate([array[:, index] for array in deltas]))
        q = float(np.quantile(absolute, .99))
        limits[gene] = q if q > 0 else .5
    frame = pd.concat(points, ignore_index=True)
    error_frame = pd.DataFrame(errors)
    profile_frame = pd.DataFrame(profiles)
    frame.to_csv(output / 'volume_plot_data.csv.gz', index=False)
    error_frame.to_csv(output / 'gene_delta_by_slice.csv', index=False)
    profile_frame.to_csv(output / 'class_marker_sums.csv', index=False)
    metadata = {'input_root': str(root.resolve()), 'genes': list(GENES),
        'sample_cap_per_slice': CAP, 'sample_seed': SEED,
        'sampling_rule': 'seed + slice ID; identical selected rows for observed and all reconstructed stacks',
        'delta': 'log1p(simulated counts) - log1p(observed counts)',
        'delta_color_limits': limits, 'delta_limit_population': 'all cells in generated targets, pooled across gaps per gene',
        'reference_anchors_in_error_statistics': False,
        'quantitative_error_summaries_use_all_cells': True,
        'donor_ids': sorted(donors), 'source_points_displayed': int((frame.gap == 0).sum())}
    (output / 'volume_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return frame, error_frame, profile_frame, metadata


def render_deltas(frame, metadata, output, dpi):
    limits = volume.spatial_limits(frame)
    fig = plt.figure(figsize=(11.6, 11.0))
    grid = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, .04],
                           left=.055, right=.91, top=.91, bottom=.05, wspace=.025, hspace=.07)
    fig.suptitle('Whole-volume expression differences', x=.055, y=.98,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(.055, .948, 'Blue: simulated lower  ·  Gray: agreement  ·  Red: simulated higher', fontsize=11)
    for row, gene in enumerate(GENES):
        bound = metadata['delta_color_limits'][gene]
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        for column, gap in enumerate(GAPS):
            axis = fig.add_subplot(grid[row, column], projection='3d')
            subset = frame[frame.gap == gap]
            anchors = subset[subset.source_kind == 'real']
            generated = subset[subset.source_kind == 'simulated']
            axis.scatter(anchors.x, anchors.y, anchors.z, color='#B7BDC4', s=.4,
                         alpha=.10, linewidths=0, depthshade=False, rasterized=True)
            zero = generated[generated[f'{gene}_delta'] == 0]
            nonzero = generated[generated[f'{gene}_delta'] != 0]
            axis.scatter(zero.x, zero.y, zero.z, color='#D4D9DF', s=.35,
                         alpha=.08, linewidths=0, depthshade=False, rasterized=True)
            axis.scatter(nonzero.x, nonzero.y, nonzero.z, c=nonzero[f'{gene}_delta'],
                         cmap=DELTA_CMAP, norm=norm, s=.6, alpha=.82, linewidths=0,
                         depthshade=False, rasterized=True)
            volume.style_3d(axis, limits, 25., -55., labels=False)
            if row == 0:
                axis.set_title(f'Gap {gap}', color=COLORS[gap], fontsize=13, pad=3)
            if column == 0:
                axis.text2D(-.05, .5, gene, transform=axis.transAxes, rotation=90,
                            ha='right', va='center', fontsize=12, fontstyle='italic', fontweight='bold')
        bar_axis = fig.add_subplot(grid[row, 3])
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=DELTA_CMAP), cax=bar_axis, extend='both')
        bar.set_label('Δ log1p counts', fontsize=10)
        bar.ax.tick_params(labelsize=9)
    fig.text(.5, .017, 'Retained real slices are context only. Error colors share the full-data ±q99 range within each gene.',
             ha='center', fontsize=9, color='#626C76')
    save(fig, output, 'study06_3d_differences', dpi)


def render_delta_profiles(errors, output, dpi):
    fig, axes = plt.subplots(4, 2, figsize=(11.6, 9.6), sharex=True)
    fig.subplots_adjust(left=.095, right=.98, top=.89, bottom=.065, hspace=.34, wspace=.23)
    fig.suptitle('Expression bias and absolute error along z', x=.095, y=.98,
                 ha='left', fontsize=18, fontweight='bold')
    for row, gene in enumerate(GENES):
        for column, metric in enumerate(('mean_delta', 'mean_absolute_delta')):
            axis = axes[row, column]
            for gap, linestyle in zip(GAPS, ('-', '--', ':')):
                data = errors[(errors.gap == gap) & (errors.gene == gene)].sort_values('z')
                axis.plot(data.z, data[metric], c=COLORS[gap], lw=1.1, ls=linestyle, label=f'Gap {gap}')
            if column == 0:
                axis.axhline(0, color='#929AA3', lw=.7)
            axis.set_ylabel(f'{gene}\nΔ log1p counts', fontsize=10)
            style_axis(axis)
            if row == 0:
                axis.set_title('Mean signed difference' if column == 0 else 'Mean absolute difference', loc='left')
            if row == 3:
                axis.set_xlabel('Target z coordinate')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(.98, .955), ncol=3)
    save(fig, output, 'study06_difference_profiles', dpi)


def render_sections(root, plan, metadata, output, dpi):
    shared = sorted(set.intersection(*(set(map(int, d['target_pool'])) for d in plan['densities'].values())))
    chosen = [shared[0], shared[len(shared)//2], shared[-1]]
    for slice_id in chosen:
        name = f'Zhuang-ABCA-1.{slice_id:03d}'
        obs, xyz, real = read_slice(Path(plan['data_dir']) / f'{name}.h5ad')
        selected = np.sort(np.random.default_rng(SEED + slice_id).choice(
            len(obs), size=min(20000, len(obs)), replace=False))
        values = {}
        records = []
        for gap in GAPS:
            density_name = next(k for k, d in plan['densities'].items() if int(d['spec']['gap']) == gap)
            gen_obs, gen_xyz, generated = read_slice(root / 'targets' / density_name / name / 'generated.h5ad')
            if not obs.index.equals(gen_obs.index) or not np.array_equal(xyz, gen_xyz):
                raise ValueError('section differences require identical observed and generated positions')
            values[gap] = np.log1p(generated[selected]) - np.log1p(real[selected])
            frame = pd.DataFrame({'gap': gap, 'slice': slice_id, 'spot_index': selected,
                'x': xyz[selected, 0], 'y': xyz[selected, 1]})
            for index, gene in enumerate(GENES):frame[gene] = values[gap][:, index]
            records.append(frame)
        pd.concat(records).to_csv(output / f'section_{slice_id:03d}_delta_data.csv.gz', index=False)
        fig = plt.figure(figsize=(11.6, 10.4))
        grid = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, .04],
            left=.06, right=.92, top=.90, bottom=.06, hspace=.12, wspace=.07)
        for row, gene in enumerate(GENES):
            bound = metadata['delta_color_limits'][gene]
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            for column, gap in enumerate(GAPS):
                axis = fig.add_subplot(grid[row, column])
                delta = values[gap][:, row]
                order = np.argsort(np.abs(delta), kind='stable')
                axis.scatter(xyz[selected[order], 0], xyz[selected[order], 1],
                    c=delta[order], cmap=DELTA_CMAP, norm=norm, s=.85,
                    alpha=.85, linewidths=0, rasterized=True)
                axis.set_aspect('equal');axis.set_axis_off()
                if row == 0:axis.set_title(f'Gap {gap}', color=COLORS[gap], fontsize=13)
                if column == 0:axis.text(-.045, .5, gene, transform=axis.transAxes,
                    rotation=90, va='center', ha='right', fontsize=12, fontstyle='italic', fontweight='bold')
            cax = fig.add_subplot(grid[row, 3])
            bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=DELTA_CMAP), cax=cax, extend='both')
            bar.set_label('Δ log1p counts', fontsize=10);bar.ax.tick_params(labelsize=9)
        fig.suptitle(f'Matched spatial differences  |  Slice {slice_id}', x=.06, y=.98,
                     ha='left', fontsize=18, fontweight='bold')
        fig.text(.06, .947, f'{len(selected):,} matched displayed cells  ·  Blue: lower simulated expression  ·  Red: higher', fontsize=10)
        fig.text(.5, .02, 'Shared full-volume color limits within each gene. Sections selected by position, independently of reconstruction performance.',
                 ha='center', fontsize=9, color='#626C76')
        save(fig, output, f'study06_delta_section_{slice_id:03d}', dpi)
    metadata['displayed_sections'] = chosen
    metadata['section_selection'] = 'first, middle and last common held-out slice by ordered position'
    metadata['section_point_cap'] = 20000
    (output / 'volume_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')


def render_marker_programs(profiles, plan, output, dpi):
    matrices, classes = [], sorted(profiles['class'].unique())
    summaries = []
    for gap in GAPS:
        target_ids = next(d['target_pool'] for d in plan['densities'].values() if int(d['spec']['gap']) == gap)
        for method_gap, method in ((0, 'Observed'), (gap, 'FEAST')):
            selected = profiles[(profiles.gap == method_gap) & profiles.target_slice.isin(target_ids)]
            table = selected.groupby(['class', 'gene'])[['count_sum', 'n_spots']].sum()
            table['mean_counts'] = table['count_sum'] / table['n_spots']
            table = table.reset_index();table['comparison_gap'] = gap;table['method'] = method
            summaries.append(table)
            matrices.append(np.log1p(table.pivot(index='class', columns='gene', values='mean_counts').reindex(index=classes, columns=GENES).to_numpy()))
    pd.concat(summaries).to_csv(output / 'matched_class_marker_means.csv', index=False)
    vmax = float(np.nanmax(np.stack(matrices)))
    fig, axes = plt.subplots(1, 6, figsize=(14., 10.2), sharey=True)
    fig.subplots_adjust(left=.19, right=.91, bottom=.115, top=.90, wspace=.12)
    for index, (axis, matrix) in enumerate(zip(axes, matrices)):
        im = axis.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
        axis.set_xticks(range(len(GENES)), GENES, rotation=55, ha='right', fontsize=10, fontstyle='italic')
        axis.set_yticks(range(len(classes)), classes, fontsize=8.5)
        axis.tick_params(length=0)
        axis.set_title(('Observed' if index % 2 == 0 else 'FEAST') + f'\nGap {GAPS[index // 2]}', fontsize=11)
        for spine in axis.spines.values():spine.set_visible(False)
    bar = fig.colorbar(im, ax=axes, location='right', fraction=.025, pad=.025)
    bar.set_label('log1p(mean counts within class)', fontsize=11)
    fig.suptitle('Preservation of supplied-class marker profiles', x=.19, y=.975, ha='left', fontsize=17, fontweight='bold')
    fig.text(.19, .94, 'Observed and FEAST profiles use the same held-out cells within each reference-density arm.', fontsize=10)
    fig.text(.5, .025, 'Class labels condition generation; this is a conditional consistency check, not independent cell-type discovery.', ha='center', fontsize=10)
    save(fig, output, 'study06_class_marker_programs', dpi)


def interactive(frame, metadata, plan, output):
    from plotly.offline import get_plotlyjs
    observed = frame[frame.gap == 0].sort_values(['target_slice', 'spot_index'])
    payload = {'x': observed.x.tolist(), 'y': observed.y.tolist(), 'z': observed.z.tolist(),
        'slices': observed.target_slice.astype(int).tolist(), 'genes': list(GENES),
        'observed': {g: observed[g].tolist() for g in GENES}, 'generated': {}, 'references': {},
        'delta_limits': metadata['delta_color_limits'], 'expression_limits': {}}
    for gap in GAPS:
        data = frame[frame.gap == gap].sort_values(['target_slice', 'spot_index'])
        payload['generated'][str(gap)] = {g: data[g].tolist() for g in GENES}
        payload['references'][str(gap)] = next(d['reference_pool'] for d in plan['densities'].values() if int(d['spec']['gap']) == gap)
    for gene in GENES:
        _, _, upper = volume.positive_norm(frame[gene].to_numpy(float));payload['expression_limits'][gene] = upper
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>Study 06 volume comparison</title>
<style>body{font:15px Arial,sans-serif;color:#24313f;margin:24px;background:white}h1{font-size:24px}select{padding:6px;margin-right:24px}#plot{height:78vh}p{max-width:1100px;color:#596571}</style>
<script>__PLOTLY__</script></head><body><h1>Study 06: matched whole-volume comparison</h1>
<label>Gene <select id="gene"></select></label><label>Reference gap <select id="gap"><option>3</option><option>5</option><option>10</option></select></label>
<label><input type="checkbox" id="section"> Single section</label>
<input type="range" id="sectionIndex" min="0" value="0"><span id="sectionLabel"></span>
<button id="view3d">3D view</button><button id="viewxy">XY view</button>
<p>Rotate any panel to synchronize the views. All 144 slice positions are represented, with the same sampled cells across methods (up to 1,200 per slice). Real anchors are gray in the difference view and excluded from error statistics.</p><div id="plot"></div>
<p>Expression: log1p counts. Differences: log1p(simulated) − log1p(observed). Blue indicates lower simulated expression, red higher. Color scales are shared across gaps within each gene; differences use the full-data ±99th percentile of absolute error.</p>
<script>const D=__DATA__; const el=document.getElementById('plot');
for(const g of D.genes){const o=document.createElement('option');o.text=g;document.getElementById('gene').add(o)}
const sliceIDs=[...new Set(D.slices)];document.getElementById('sectionIndex').max=sliceIDs.length-1;
const limits=['x','y','z'].map(k=>{let lo=Infinity,hi=-Infinity;for(const v of D[k]){lo=Math.min(lo,v);hi=Math.max(hi,v)}return [lo,hi]});
const scene=(domain)=>({domain:{x:domain,y:[0,1]},aspectmode:'data',xaxis:{title:'x',range:limits[0]},yaxis:{title:'y',range:limits[1]},zaxis:{title:'z',range:limits[2]},camera:{eye:{x:1.5,y:1.5,z:.8}}});
const base={type:'scatter3d',mode:'markers',showlegend:false,hoverinfo:'skip'};
function draw(){const gene=document.getElementById('gene').value,gap=document.getElementById('gap').value;
const real=D.observed[gene].map(Math.log1p),gen=D.generated[gap][gene].map(Math.log1p),refs=new Set(D.references[gap]),a=[],s=[];
const single=document.getElementById('section').checked,section=sliceIDs[+document.getElementById('sectionIndex').value],all=[];
document.getElementById('sectionLabel').textContent=' Slice '+section+' ';
for(let i=0;i<D.x.length;i++){if(single&&D.slices[i]!==section)continue;all.push(i);(refs.has(D.slices[i])?a:s).push(i)}
const subset=idx=>({x:idx.map(i=>D.x[i]),y:idx.map(i=>D.y[i]),z:idx.map(i=>D.z[i])});
const expression=(v,sc,bar)=>{const positive=all.filter(i=>v[i]>0),zero=all.filter(i=>v[i]===0);return [{...base,...subset(zero),scene:sc,marker:{size:1,opacity:.05,color:'#BAC1C9'}},{...base,...subset(positive),scene:sc,marker:{size:1.5,opacity:.8,color:positive.map(i=>v[i]),colorscale:'Viridis',cmin:0,cmax:D.expression_limits[gene],showscale:bar,colorbar:{x:.315,len:.65,thickness:10,title:{text:'log1p counts'}}}}]};
const differences=s.filter(i=>gen[i]!==real[i]),gray=a.concat(s.filter(i=>gen[i]===real[i]));
const traces=[...expression(real,'scene',true),...expression(gen,'scene2',false),
{...base,...subset(gray),scene:'scene3',marker:{size:1.1,opacity:.08,color:'#AAB2BB'}},
{...base,...subset(differences),scene:'scene3',marker:{size:1.5,opacity:.82,color:differences.map(i=>gen[i]-real[i]),colorscale:[[0,'#2166AC'],[.25,'#67A9CF'],[.5,'#E0E0E0'],[.75,'#EF8A62'],[1,'#B2182B']],cmin:-D.delta_limits[gene],cmax:D.delta_limits[gene],showscale:true,colorbar:{x:1.02,len:.65,thickness:10,title:{text:'Δ log1p counts'}}}}];
Plotly.react(el,traces,{margin:{l:0,r:80,t:40,b:0},font:{family:'Arial'},uirevision:'same-volume',scene:scene([0,.3]),scene2:scene([.35,.65]),scene3:scene([.7,1]),annotations:[{x:.15,y:1.05,text:'Observed',showarrow:false,xref:'paper',yref:'paper'},{x:.5,y:1.05,text:'FEAST · gap '+gap,showarrow:false,xref:'paper',yref:'paper'},{x:.85,y:1.05,text:single&&s.length===0?'Retained reference; no generated cells':'Generated − observed',showarrow:false,xref:'paper',yref:'paper'}]},{responsive:true,displaylogo:false});}
document.getElementById('gene').onchange=draw;document.getElementById('gap').onchange=draw;draw();
document.getElementById('section').onchange=draw;document.getElementById('sectionIndex').oninput=draw;
for(const [id,eye] of [['view3d',{x:1.5,y:1.5,z:.8}],['viewxy',{x:0,y:0,z:2.5}]])document.getElementById(id).onclick=()=>Plotly.relayout(el,{'scene.camera':{eye},'scene2.camera':{eye},'scene3.camera':{eye}});
let synchronizing=false;el.on('plotly_relayout',event=>{if(synchronizing)return;const key=Object.keys(event).find(k=>/^scene[23]?\\.camera$/.test(k));if(!key)return;const updates={};for(const s of ['scene','scene2','scene3'])if(s+'.camera'!==key)updates[s+'.camera']=event[key];synchronizing=true;Plotly.relayout(el,updates).finally(()=>synchronizing=false)});
</script></body></html>'''
    html = template.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', json.dumps(payload, separators=(',', ':')))
    (output / 'study06_volume_explorer.html').write_text(html)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'figures/whole_volume_1.0.6')
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    plan,validation=volume.require_plan(args.input_root)
    if args.render_only:
        frame=pd.read_csv(args.output_dir/'volume_plot_data.csv.gz')
        errors=pd.read_csv(args.output_dir/'gene_delta_by_slice.csv');profiles=pd.read_csv(args.output_dir/'class_marker_sums.csv')
        metadata=json.loads((args.output_dir/'volume_provenance.json').read_text())
    else:frame,errors,profiles,metadata=collect(args.input_root,plan,args.output_dir)
    volume.GAP_LABELS.update({int(d['spec']['gap']):f"Gap {d['spec']['gap']} ({len(d['reference_pool'])} real / {len(d['target_pool'])} generated)" for d in plan['densities'].values()})
    volume.configure_matplotlib()
    volume.render_volume(frame,args.output_dir/'study06_3d_volume',args.dpi)
    volume.render_genes(frame,args.output_dir/'study06_3d_expression',args.dpi)
    configure_style();render_deltas(frame,metadata,args.output_dir,args.dpi)
    render_delta_profiles(errors,args.output_dir,args.dpi);render_sections(args.input_root,plan,metadata,args.output_dir,args.dpi)
    render_marker_programs(profiles,plan,args.output_dir,args.dpi)
    interactive(frame,metadata,plan,args.output_dir)
    print(f'Completed whole-volume figures and offline explorer: {args.output_dir}',flush=True)

if __name__=='__main__':main()
