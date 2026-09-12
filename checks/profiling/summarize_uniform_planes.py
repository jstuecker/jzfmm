"""Summarize actual GPU durations, never host-side CUDA launch durations."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--step-name', default='full_force')
    args = parser.parse_args()
    root = args.directory
    timing = json.loads((root/'timings.json').read_text())
    path = next((root/'trace').rglob('*.trace.json.gz'))
    with gzip.open(path) as f:
        events = json.load(f)['traceEvents']
    gpu_pids = {e['pid'] for e in events if e.get('name') == 'process_name' and '/device:GPU:' in e.get('args', {}).get('name', '')}
    assert len(gpu_pids) == 1, 'This analysis is for one GPU'
    gpu = sorted([e for e in events if e.get('pid') in gpu_pids and e.get('ph') == 'X'], key=lambda e:e['ts'])
    steps = sorted([e for e in events if e.get('name') == args.step_name and e.get('ph') == 'X'], key=lambda e:e['ts'])
    assert steps and gpu, 'Missing CUDA trace or step annotations'
    rows = []
    details = []
    for step in steps:
        start, end = step['ts'], step['ts'] + step['dur']
        ev = [e for e in gpu if start <= e['ts'] and e['ts']+e['dur'] <= end]
        # Summing event durations assumes this single compute stream is serial.
        compute_tids = {e['tid'] for e in ev if 'kernel_details' in e.get('args', {})}
        assert len(compute_tids) == 1
        kernels = [e for e in ev if e['tid'] in compute_tids]
        assert all(a['ts']+a['dur'] <= b['ts']+0.01 for a,b in zip(kernels,kernels[1:])), 'Overlapping GPU work'
        count = [i for i,e in enumerate(ev) if 'void CountInteractionsAndM2L<' in e['name']]
        insert = [i for i,e in enumerate(ev) if 'void InsertInteractions<' in e['name']]
        leaf = [e for e in ev if 'void LeafLeafPairSummation<' in e['name']]
        assert len(count) == len(insert) == timing['num_planes'] and len(leaf) == 1
        row = {'step': int(step['args']['step_num']), 'profiled_wall_ms': step['dur']/1000,
               'leaf_leaf_ms': leaf[0]['dur']/1000}
        translations = {i for i,e in enumerate(ev) if 'void TranslateLocalToLocal<' in e['name']}
        translation_events = translations | {i-1 for i in translations if i > 0 and ev[i-1]['name'].startswith('Memset')}
        assigned = set()
        for plane, c, ins in zip(reversed(range(timing['num_planes'])), count, insert):
            assert c < ins and (ins < count[count.index(c)+1] if c != count[-1] else True)
            # Include the count-array memset belonging to the same FFI call.
            begin = c
            if begin > 0 and ev[begin-1]['name'].startswith('Memset'):
                begin -= 1
            stage = set(range(begin, ins+1)) - translation_events
            assert not (assigned & stage), 'Overlapping plane accounting'
            assigned |= stage
            row[f'plane_{plane}_ms'] = sum(ev[i]['dur'] for i in sorted(stage))/1000
            details.append({'step':row['step'], 'plane':plane,
                'm2l_count_ms':ev[c]['dur']/1000, 'insert_ms':ev[ins]['dur']/1000,
                'total_ms':row[f'plane_{plane}_ms']})
        row['interaction_sum_ms'] = row['leaf_leaf_ms'] + sum(row[f'plane_{p}_ms'] for p in range(timing['num_planes']))
        row['gpu_total_ms'] = sum(e['dur'] for e in ev)/1000
        row['other_gpu_ms'] = sum(e['dur'] for e in ev)/1000 - row['leaf_leaf_ms'] - sum(row[f'plane_{p}_ms'] for p in range(timing['num_planes']))
        assert row['other_gpu_ms'] >= -1e-8, 'Double-counted GPU events'
        row['host_gaps_ms'] = row['profiled_wall_ms'] - sum(e['dur'] for e in ev)/1000
        rows.append(row)
    for name, data in [('trace_samples.csv',rows), ('plane_details.csv',details)]:
        with (root/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
    summary={k:{'median':statistics.median(r[k] for r in rows), 'min':min(r[k] for r in rows), 'max':max(r[k] for r in rows)} for k in rows[0] if k!='step'}
    (root/'trace_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
