"""Execute the authorized fresh local-generative Studies 06/07 with two workers."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / 'envs/feast-study06-log-1.0.5-cu124/bin/python'
LOGS = ROOT / '.work/generative_3d_v1'
OUT06 = ROOT / '06_3d_stack/outputs/generative_five_reference_v1'
OUT07 = ROOT / '07_3d_transfer/outputs/generative_transfer_v1'
ENV = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           NUMEXPR_NUM_THREADS='2', MPLCONFIGDIR='/tmp/feast-generative-mpl')
ENV.pop('PYTHONPATH', None)


def run(name, *args):
    log = LOGS / f'{name}.log'
    print(f'start {name}: {log}', flush=True)
    with log.open('a') as handle:
        subprocess.run([str(PYTHON), *map(str, args)], cwd=ROOT, env=ENV,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f'complete {name}', flush=True)


def calibrate07():
    for age in ['E15.5', 'E18.5']:
        run(f'07_calibrate_{age}', '07_3d_transfer/calibrate.py', '--age', age)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['06', '07', 'both'], default='both')
    parser.add_argument('--after-calibration', action='store_true')
    args = parser.parse_args()
    study06 = args.study in ('06', 'both')
    study07 = args.study in ('07', 'both')
    LOGS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as workers:
        if not args.after_calibration:
            jobs = []
            if study06:
                jobs.append(workers.submit(run, '06_preflight', '06_3d_stack/preflight.py',
                    '--data-dir', ROOT.parent / 'Datasets/Processed/Allen_Zhuang_ABCA_1/h5ad', '--output-dir', OUT06))
            if study07:
                jobs.append(workers.submit(calibrate07))
            for job in jobs:
                job.result()
        # One representative full target per density and age, before production.
        jobs = []
        if study06:
            jobs += [workers.submit(run, f'06_representative_gap{gap}', '06_3d_stack/run.py',
                '--output-dir', OUT06, '--gap', gap, '--target-id', 5) for gap in [3, 5, 10]]
        if study07:
            jobs += [workers.submit(run, f'07_representative_{age}', '07_3d_transfer/run.py',
                 '--age', age, '--shard-id', 'representative', '--z-indices', 0)
                 for age in ['E15.5', 'E18.5']]
        for job in jobs:
            job.result()
        if study06:
            run('06_representative_validation', '06_3d_stack/validate.py', '--output-dir', OUT06,
                '--canary-target', '3:5', '--canary-target', '5:5', '--canary-target', '10:5')
        jobs = []
        if study06:
            jobs += [workers.submit(run, f'06_gap{gap}_shard{shard}', '06_3d_stack/run.py',
                '--output-dir', OUT06, '--gap', gap, '--shard-index', shard, '--shard-count', 2)
                for gap in [3, 5, 10] for shard in [0, 1]]
        for age, stop in ([('E15.5', 158), ('E18.5', 202)] if study07 else []):
            midpoint = stop // 2
            for shard, start, end in [(0, 1, midpoint), (1, midpoint, stop)]:
                jobs.append(workers.submit(run, f'07_{age}_shard{shard}', '07_3d_transfer/run.py',
                    '--age', age, '--shard-id', f'production_{shard}', '--start-index', start, '--stop-index', end))
        for job in jobs:
            job.result()
    if study06:
        run('06_validate', '06_3d_stack/validate.py', '--output-dir', OUT06)
        run('06_evaluate', '06_3d_stack/aggregate.py', '--output-dir', OUT06)
        run('06_plot', 'visualization/06_3d_stack/plot.py')
    if study07:
        for age in ['E15.5', 'E18.5']:
            run(f'07_consolidate_{age}', '07_3d_transfer/consolidate.py', '--age', age)
        run('07_validate', '07_3d_transfer/validate.py')
        run('07_evaluate', '07_3d_transfer/evaluate.py')
        run('07_plot', 'visualization/07_3d_transfer/plot.py')
    if study06 and study07:
        run('generation_report', 'scripts/report_generative_3d.py')
    count = (336 if study06 else 0) + (360 if study07 else 0)
    print(f'All {count} selected targets, validation, evaluation and figures completed.', flush=True)

if __name__ == '__main__':
    main()
