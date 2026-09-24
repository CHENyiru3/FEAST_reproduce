"""Observed, FEAST and native SpatialZ section maps and whole-volume views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_3d_gene_distribution as volume
from plot_results import DEFAULT_INPUT, DEFAULT_SPATIALZ, configure_style, marker_maps, save
from plot_volume_differences import read_slice

GENES = volume.GENES
GAPS = (3, 5, 10)
METHODS = ('Observed', 'FEAST', 'SpatialZ')
PANELS = [('Observed', 0)] + [(method, gap) for gap in GAPS for method in METHODS[1:]]


def collect(feast_root, spatialz_root, plan, output):
    points, profiles = [], []
    arms = {int(d['spec']['gap']): (name, set(map(int, d['reference_pool']))) for name, d in plan['densities'].items()}
    for number, sid in enumerate(map(int, plan['available_slices']), start=1):
        name = f'Zhuang-ABCA-1.{sid:03d}'
        real_obs, real_xyz, real_values = read_slice(Path(plan['data_dir']) / f'{name}.h5ad')
        for method, gap in PANELS:
            is_reference = gap == 0 or sid in arms[gap][1]
            if is_reference:
                obs, xyz, values = real_obs, real_xyz, real_values
            else:
                root = feast_root if method == 'FEAST' else spatialz_root
                obs, xyz, values = read_slice(root / 'targets' / arms[gap][0] / name / 'generated.h5ad')
                if method == 'FEAST' and (not obs.index.equals(real_obs.index) or not np.array_equal(xyz, real_xyz)):
                    raise ValueError(f'{name}, gap {gap}: FEAST no longer matches the target cells')
            selected = volume.sampled_indices(len(obs), 2026 + sid)
            frame = pd.DataFrame({'method': method, 'gap': gap, 'target_slice': sid,
                                  'spot_index': selected, 'x': xyz[selected, 0],
                                  'y': xyz[selected, 1], 'z': xyz[selected, 2],
                                  'source_kind': 'real' if is_reference else 'simulated'})
            labels, codes = np.unique(obs['class'].astype(str), return_inverse=True)
            sizes = np.bincount(codes)
            for j, gene in enumerate(GENES):
                frame[gene] = values[selected, j]
                sums = np.bincount(codes, weights=values[:, j])
                for label, size, total in zip(labels, sizes, sums):
                    profiles.append({'method': method, 'gap': gap, 'target_slice': sid,
                                     'class': label, 'gene': gene, 'n_cells': int(size),
                                     'count_sum': float(total), 'source_kind': 'real' if is_reference else 'simulated'})
            points.append(frame)
        if number % 12 == 0:
            print(f'Collected {number}/144 positions for Observed, FEAST and SpatialZ.', flush=True)
    frame = pd.concat(points, ignore_index=True)
    profiles = pd.DataFrame(profiles)
    frame.to_csv(output / 'native_volume_plot_data.csv.gz', index=False)
    profiles.to_csv(output / 'native_class_marker_sums.csv', index=False)
    return frame, profiles


def render_volume(frame, output, dpi):
    limits = volume.spatial_limits(frame)
    colors = {gene: volume.positive_norm(frame[gene].to_numpy(float))[1] for gene in GENES}
    fig = plt.figure(figsize=(19.5, 11.5))
    grid = fig.add_gridspec(4, 8, width_ratios=[1]*7+[.035], left=.025, right=.95,
                          top=.90, bottom=.065, hspace=.12, wspace=.02)
    for row, gene in enumerate(GENES):
        for col, (method, gap) in enumerate(PANELS):
            ax = fig.add_subplot(grid[row, col], projection='3d')
            subset = frame[(frame.method == method) & (frame.gap == gap)]
            display = np.log1p(subset[gene].to_numpy(float))
            ax.scatter(subset.x, subset.y, subset.z, color='#CCD3D9', s=.35,
                       alpha=.08, linewidths=0, depthshade=False, rasterized=True)
            for source, marker, size in [('real', '^', .65), ('simulated', 'o', .45)]:
                positions = np.flatnonzero((subset.source_kind.to_numpy() == source) & (display > 0))
                positions = positions[np.argsort(display[positions], kind='stable')]
                selected = subset.iloc[positions]
                ax.scatter(selected.x, selected.y, selected.z, c=display[positions],
                           cmap='viridis', norm=colors[gene], s=size, marker=marker,
                           alpha=.85, linewidths=0, depthshade=False, rasterized=True)
            volume.style_3d(ax, limits, 25., -55., labels=False)
            if row == 0:
                ax.set_title(method if gap == 0 else f'{method}\ngap {gap}', fontsize=11)
            if col == 0:
                ax.text2D(-.04, .5, gene, transform=ax.transAxes, rotation=90,
                          ha='right', va='center', fontsize=12, fontstyle='italic')
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=colors[gene], cmap='viridis'), cax=fig.add_subplot(grid[row, 7]))
        bar.set_label('log1p counts', fontsize=10)
    fig.suptitle('Observed, FEAST and SpatialZ expression across all 144 slice positions', fontsize=18, x=.035, y=.98, ha='left')
    fig.text(.035, .942, 'Native coordinates for each method; shared spatial limits and gene-wise color scales. Retained reference slices are included.', fontsize=11)
    fig.text(.035, .02, 'Up to 1,200 cells per slice and method. Triangles: real cells; circles: generated cells. Colors saturate at the pooled positive 99th percentile.', fontsize=10)
    save(fig, output, 'study06_native_3d_expression', dpi)
    return {gene: float(norm.vmax) for gene, norm in colors.items()}


def render_class_profiles(profiles, plan, output, dpi):
    classes = sorted(profiles['class'].unique())
    matrices, records = [], []
    for gap in GAPS:
        targets = next(d['target_pool'] for d in plan['densities'].values() if int(d['spec']['gap']) == gap)
        for method in METHODS:
            selected = profiles[(profiles.method == method) & (profiles.gap == (0 if method == 'Observed' else gap)) & profiles.target_slice.isin(targets)]
            table = selected.groupby(['class', 'gene'])[['count_sum', 'n_cells']].sum()
            table['mean_counts'] = table.count_sum / table.n_cells
            records.append(table.reset_index().assign(method=method, comparison_gap=gap))
            matrices.append(np.log1p(table.mean_counts.unstack().reindex(index=classes, columns=GENES)))
    pd.concat(records).to_csv(output / 'native_class_marker_means.csv', index=False)
    vmax = float(np.nanmax(np.stack(matrices)))
    cmap = mpl.colormaps['viridis'].copy(); cmap.set_bad('#DDDDDD')
    fig, axes = plt.subplots(1, 9, figsize=(17, 10.5), sharey=True)
    fig.subplots_adjust(left=.17, right=.93, bottom=.12, top=.87, wspace=.15)
    for i, (ax, matrix) in enumerate(zip(axes, matrices)):
        im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(4), GENES, rotation=60, ha='right', fontstyle='italic', fontsize=9)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_title(f'{METHODS[i%3]}\nGap {GAPS[i//3]}', fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    bar = fig.colorbar(im, ax=axes, fraction=.018, pad=.015)
    bar.set_label('log1p(mean counts within class)')
    fig.suptitle('Class-marker profiles on held-out slices', x=.17, y=.97, ha='left', fontsize=17)
    fig.text(.17, .93, 'Each method uses its own cells and labels. Counts are pooled within class over the same held-out slice IDs.', fontsize=10)
    fig.text(.17, .03, 'All cells contribute to these means. Gray: class absent. FEAST receives target labels; SpatialZ infers its labels from references.', fontsize=10)
    save(fig, output, 'study06_native_class_marker_profiles', dpi)


def interactive(frame, plan, limits, output):
    from plotly.offline import get_plotlyjs
    stacks = {}
    for method, gap in PANELS:
        subset = frame[(frame.method == method) & (frame.gap == gap)]
        stacks[f'{method}_{gap}'] = {key: subset[key].round(6).tolist() for key in ['x', 'y', 'z', *GENES]}
        stacks[f'{method}_{gap}']['slice'] = subset.target_slice.astype(int).tolist()
    payload = {'stacks': stacks, 'genes': list(GENES), 'limits': limits,
               'slices': list(map(int, plan['available_slices'])),
               'references': {str(d['spec']['gap']): d['reference_pool'] for d in plan['densities'].values()},
               'xyz_limits': [[float(frame[k].min()), float(frame[k].max())] for k in ['x','y','z']]}
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>Study 06 native volume comparison</title>
<style>body{font:15px Arial,sans-serif;margin:24px;color:#24313f}select,button{padding:6px;margin-right:16px}#plot{height:78vh}p{max-width:1200px}</style>
<script>__PLOTLY__</script></head><body><h1>Observed · FEAST · SpatialZ</h1>
<label>Gene <select id="gene"></select></label><label>Reference gap <select id="gap"><option>3</option><option>5</option><option>10</option></select></label>
<label><input type="checkbox" id="single"> Single slice</label><input type="range" id="slice" min="0" value="0"><span id="sliceLabel"></span>
<label><input type="checkbox" id="anchors" checked> Include retained real slices</label>
<button id="xy">XY view</button><button id="volume">3D view</button>
<p>Each method uses its own coordinates and cells. Colors are shared across methods and gaps within each gene. Rotate a panel to synchronize all views.</p><div id="plot"></div>
<p>Up to 1,200 sampled cells per slice and method; log1p raw counts. Whole stacks include real anchors when selected. This view compares expression patterns without matching individual cells.</p>
<script>const D=__DATA__,el=document.getElementById('plot'),names=['Observed','FEAST','SpatialZ'];
for(const gene of D.genes){const option=document.createElement('option');option.text=gene;document.getElementById('gene').add(option)}
document.getElementById('slice').max=D.slices.length-1;
const scene=domain=>({domain:{x:domain,y:[0,1]},aspectmode:'data',xaxis:{title:'x',range:D.xyz_limits[0]},yaxis:{title:'y',range:D.xyz_limits[1]},zaxis:{title:'z',range:D.xyz_limits[2]},camera:{eye:{x:1.5,y:1.5,z:.8}}});
function draw(){const gene=document.getElementById('gene').value,gap=document.getElementById('gap').value,single=document.getElementById('single').checked,slice=D.slices[+document.getElementById('slice').value],anchors=document.getElementById('anchors').checked,refs=new Set(D.references[gap]);
document.getElementById('sliceLabel').textContent=' Slice '+slice+' ';const traces=[];
for(let m=0;m<3;m++){const data=D.stacks[names[m]+'_'+(m===0?0:gap)],sceneName=m===0?'scene':'scene'+(m+1),positive=[],zero=[];
for(let i=0;i<data.x.length;i++){if(single&&data.slice[i]!==slice)continue;if(!anchors&&refs.has(data.slice[i]))continue;(data[gene][i]>0?positive:zero).push(i)}
const xyz=idx=>({x:idx.map(i=>data.x[i]),y:idx.map(i=>data.y[i]),z:idx.map(i=>data.z[i])});
traces.push({type:'scatter3d',mode:'markers',scene:sceneName,...xyz(zero),showlegend:false,hoverinfo:'skip',marker:{color:'#BBC3CB',size:1,opacity:.06}});
traces.push({type:'scatter3d',mode:'markers',scene:sceneName,...xyz(positive),showlegend:false,hoverinfo:'skip',marker:{color:positive.map(i=>Math.log1p(data[gene][i])),size:1.5,opacity:.8,colorscale:'Viridis',cmin:0,cmax:D.limits[gene],showscale:m===2,colorbar:{title:{text:'log1p counts'},thickness:12,len:.7}}});}
Plotly.react(el,traces,{margin:{l:0,r:85,t:40,b:0},font:{family:'Arial'},uirevision:'native-volume',scene:scene([0,.30]),scene2:scene([.35,.65]),scene3:scene([.70,1]),annotations:names.map((name,i)=>({x:[.15,.50,.85][i],y:1.03,text:name+(i?' · gap '+gap:''),showarrow:false,xref:'paper',yref:'paper'}))},{responsive:true,displaylogo:false});}
for(const id of ['gene','gap','single','anchors'])document.getElementById(id).onchange=draw;
document.getElementById('slice').oninput=draw;
for(const [id,eye] of [['xy',{x:0,y:0,z:2.5}],['volume',{x:1.5,y:1.5,z:.8}]])document.getElementById(id).onclick=()=>Plotly.relayout(el,{'scene.camera':{eye},'scene2.camera':{eye},'scene3.camera':{eye}});
draw();let syncing=false;el.on('plotly_relayout',event=>{if(syncing)return;const key=Object.keys(event).find(k=>/^scene[23]?\\.camera$/.test(k));if(!key)return;const changes={};for(const name of ['scene','scene2','scene3'])if(name+'.camera'!==key)changes[name+'.camera']=event[key];syncing=true;Plotly.relayout(el,changes).finally(()=>syncing=false)});
</script></body></html>'''
    html = template.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', json.dumps(payload, separators=(',',':')))
    (output / 'study06_native_volume_explorer.html').write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--spatialz-root', type=Path, default=DEFAULT_SPATIALZ)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'figures/spatialz_native_views_v1')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    plan, _ = volume.require_plan(args.input_root)
    evaluation = json.loads((args.spatialz_root / 'evaluation/evaluation.json').read_text())
    if evaluation['status'] != 'completed' or evaluation['targets_evaluated'] != 336 or Path(evaluation['feast_root']).resolve() != args.input_root.resolve():
        raise ValueError('SpatialZ must have completed evaluation against this FEAST run')
    if args.render_only:
        frame = pd.read_csv(args.output_dir / 'native_volume_plot_data.csv.gz')
        profiles = pd.read_csv(args.output_dir / 'native_class_marker_sums.csv')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        frame, profiles = collect(args.input_root, args.spatialz_root, plan, args.output_dir)
    configure_style()
    shared = sorted(set.intersection(*(set(map(int,d['target_pool'])) for d in plan['densities'].values())))
    maps = [marker_maps(plan, args.input_root, args.output_dir, args.dpi, args.spatialz_root, sid)
            for sid in (shared[0], shared[len(shared)//2], shared[-1])]
    print('Rendered native section maps.', flush=True)
    limits = render_volume(frame, args.output_dir, args.dpi)
    print('Rendered native volume expression.', flush=True)
    render_class_profiles(profiles, plan, args.output_dir, args.dpi)
    interactive(frame, plan, limits, args.output_dir)
    record = {'input_root': str(args.input_root.resolve()), 'spatialz_root': str(args.spatialz_root.resolve()),
              'entry_point': str(Path(__file__).resolve()), 'genes': list(GENES), 'section_maps': maps,
              'sample_cap_per_slice': volume.SAMPLE_CAP_PER_SLICE, 'sample_seed': '2026 + slice ID, applied within each method',
              'volume_color_limits': limits, 'volume_color_rule': 'pooled positive log1p-count 99th percentile on displayed cells',
              'class_means': 'all cells; each method uses its own class labels; same held-out slice IDs per gap',
              'anchors_in_volume': True, 'spatialz_cell_matching': False, 'png_dpi': args.dpi}
    (args.output_dir / 'figure_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    (args.output_dir / 'CAPTIONS.md').write_text(
        '# Native SpatialZ comparison views\n\n'
        'Observed, FEAST and SpatialZ are displayed on their own coordinates, with shared axes and pooled gene-wise color limits. '
        'Slc17a7, Gad2, Gfap and Reln are the existing marker panel. Sections 5, 78 and 149 are the first, middle and last common held-out slices, selected by position. '
        'Section maps show every cell; the volume and offline explorer show up to 1,200 cells per slice and method across all 144 levels, including retained real anchors. '
        'Counts use log1p for display, with the pooled positive 99th percentile as the upper color limit; only display colors saturate. '
        'Class-marker heatmaps use all cells from held-out slices and each method’s own labels; absent classes are gray. '
        'SpatialZ cells are never matched to observed cells, so no cellwise difference map is computed for SpatialZ. '
        'These are descriptive views, not newly computed anatomical-fidelity or z-continuity metrics.\n')
    print(f'Completed native SpatialZ views: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
