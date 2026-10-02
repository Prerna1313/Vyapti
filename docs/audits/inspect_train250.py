"""Read-only audit of current development data; prints evidence, never edits inputs."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import sys
import time

import h5py
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    raw = root / 'Data'
    cache = root / 'runs/tsrd_cache/train_250'
    if '--recheck-modes' in sys.argv:
        output = Path(__file__).with_name('train250-data-verification.json')
        result = json.loads(output.read_text())
        result['errors'] = [e for e in result['errors'] if e[1] != 'H5 mode/path mismatch']
        for path in raw.rglob('*.h5'):
            with h5py.File(path) as h:
                mode = h['metadata/receiver'].attrs['scan_mode']
                mode = mode.decode() if isinstance(mode, bytes) else str(mode)
            normalized = {'scanning': 'scan', 'stare': 'stare'}.get(mode.lower(), mode.lower())
            if normalized != path.parent.name.split('_')[-1]:
                result['errors'].append([path.relative_to(raw).as_posix(), 'H5 mode/path mismatch'])
        result['mode_recheck_note'] = 'Rechecked all 379 H5 modes after correcting audit checker to accept the dataset value Scanning as scan. Original pulse comparisons and hashes retained.'
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'errors': result['errors'], 'note': result['mode_recheck_note']}))
        return
    manifest = json.loads((cache / 'manifest.json').read_text())
    selected = json.loads((cache / 'selected_config_ids.json').read_text())['selected_config_ids']
    entries = json.loads((cache / 'emitter_index.json').read_text())['entries']
    summaries = json.loads((cache / 'config_summaries.json').read_text())['configs']
    historic = json.loads((root / 'data_provenance/tsrd_corpus_manifest.json').read_text())
    old_hashes = {x['path']: x['sha256'] for x in historic['files']}
    by_config = defaultdict(list)
    for entry in entries:
        by_config[entry['config_id']].append(entry)
    summary_by_id = {x['config_id']: x for x in summaries}
    result = {
        'cache_manifest': {k: v for k, v in manifest.items() if k != 'selected_config_ids'},
        'counts': {}, 'split_ids': {}, 'errors': [], 'hash_mismatches': [],
        'raw_file_hashes': {}, 'cache_file_hashes': {}, 'receiver_geometries': {},
        'train_pulses_checked': 0, 'train_emitters_checked': 0,
        'nonfinite_values': 0, 'outside_mission_pulses': 0,
        'raw_emitter_sequences_requiring_sort': 0, 'npz_unsorted_sequences': 0,
    }
    for mode in ('stare', 'scan'):
        for split in ('train', 'val', 'test'):
            paths = sorted((raw / mode / f'{split}_{mode}').glob('*.h5'))
            key = f'{mode}/{split}'
            result['counts'][key] = len(paths)
            result['split_ids'][key] = [p.stem for p in paths]
    npz_ids = {p.stem for p in (cache / 'configs').glob('*.npz')}
    raw_ids = set(result['split_ids']['stare/train'])
    result['selection_consistency'] = {
        'selection_unique': len(selected) == len(set(selected)),
        'manifest_matches_selection': manifest['selected_config_ids'] == selected,
        'raw_matches_selection': raw_ids == set(selected),
        'npz_matches_selection': npz_ids == set(selected),
        'index_matches_selection': set(by_config) == set(selected),
        'summaries_match_selection': set(summary_by_id) == set(selected),
        'index_uids_unique': len({e['uid'] for e in entries}) == len(entries),
        'index_count': len(entries),
        'unresolved_stare_references': sum(not (raw / e['stare_file']).is_file() for e in entries),
        'unresolved_npz_references': sum(not (cache / e['npz_file']).is_file() for e in entries),
        'extra_raw_ids': sorted(raw_ids - set(selected)),
        'missing_raw_ids': sorted(set(selected) - raw_ids),
    }
    print('INITIAL ' + json.dumps({k: result[k] for k in ('counts', 'selection_consistency')}), flush=True)
    started = time.monotonic()
    geometry = Counter()
    for n, path in enumerate(sorted(raw.rglob('*.h5'))):
        rel = path.relative_to(raw).as_posix()
        digest = sha256(path)
        result['raw_file_hashes'][rel] = digest
        if rel in old_hashes and digest != old_hashes[rel]:
            result['hash_mismatches'].append(rel)
        with h5py.File(path) as h:
            if h['data'].ndim != 2 or h['data'].shape[1] != 5 or h['data'].shape[0] != h['labels'].shape[0]:
                result['errors'].append([rel, 'H5 shape mismatch'])
            receiver = h['metadata/receiver']
            mode = receiver.attrs.get('scan_mode', '')
            mode = mode.decode() if isinstance(mode, bytes) else str(mode)
            normalized_mode = {'scanning': 'scan', 'stare': 'stare'}.get(mode.lower(), mode.lower())
            if normalized_mode != path.parent.name.split('_')[-1]:
                result['errors'].append([rel, 'H5 mode/path mismatch'])
            contract = {'mode': mode, 'halfwidth_mhz': float(receiver.attrs['bandwith_mhz']),
                        'collection_s': float(receiver.attrs['collection_time_s']),
                        'centres_mhz': receiver['dwell_centres_mhz'][:].tolist()}
            geometry[json.dumps(contract, sort_keys=True)] += 1
        if (n + 1) % 50 == 0:
            print(f'H5_HASH_PROGRESS {n + 1} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    result['receiver_geometries'] = [{'files': v, **json.loads(k)} for k, v in geometry.items()]
    hash_groups = defaultdict(list)
    for rel, digest in result['raw_file_hashes'].items():
        hash_groups[digest].append(rel)
    result['cross_split_identical_files'] = [v for v in hash_groups.values()
        if len({p.split('/')[1].split('_')[0] for p in v}) > 1]
    result['historic_hashes_compared'] = sum(p in old_hashes for p in result['raw_file_hashes'])
    fields = ['pdw_toa_us_', 'pdw_frequency_mhz_', 'pdw_pulse_width_', 'pdw_aoa_deg_', 'pdw_amplitude_dbm_']
    for n, cid in enumerate(selected):
        hpath = raw / 'stare/train_stare' / f'{cid}.h5'
        zpath = cache / 'configs' / f'{cid}.npz'
        if not hpath.is_file() or not zpath.is_file():
            result['errors'].append([cid, 'missing input'])
            continue
        result['cache_file_hashes'][zpath.name] = sha256(zpath)
        with h5py.File(hpath) as h:
            data = h['data'][:]
            labels = h['labels'][:].reshape(-1)
            horizon_us = float(h['metadata/receiver'].attrs['collection_time_s']) * 1e6
        result['nonfinite_values'] += int(np.count_nonzero(~np.isfinite(data)))
        result['outside_mission_pulses'] += int(np.count_nonzero((data[:, 0] < 0) | (data[:, 0] >= horizon_us)))
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(labels, kind='stable')
        starts = np.cumsum(np.r_[0, counts])
        if len(data) != summary_by_id[cid]['n_pdws'] or len(unique) != summary_by_id[cid]['n_emitters']:
            result['errors'].append([cid, 'summary count mismatch'])
        indexed = {int(e['source_label']): e for e in by_config[cid]}
        with np.load(zpath, allow_pickle=False) as z:
            if set(z['emitter_labels'].tolist()) != set(unique.tolist()) or set(indexed) != set(unique.tolist()):
                result['errors'].append([cid, 'label set mismatch'])
            for j, label in enumerate(unique):
                expected = data[order[starts[j]:starts[j+1]]]
                if np.any(np.diff(expected[:, 0]) < 0):
                    result['raw_emitter_sequences_requiring_sort'] += 1
                expected = expected[np.argsort(expected[:, 0], kind='stable')]
                suffix = str(int(label))
                for column, prefix in enumerate(fields):
                    actual = z[prefix + suffix]
                    if column == 0 and np.any(np.diff(actual) < 0):
                        result['npz_unsorted_sequences'] += 1
                    if not np.array_equal(actual, expected[:, column], equal_nan=True):
                        result['errors'].append([cid, int(label), prefix, 'source mismatch'])
                if int(z['total_pulses_' + suffix]) != len(expected) or indexed[int(label)]['recorded_pulse_count'] != len(expected):
                    result['errors'].append([cid, int(label), 'pulse count mismatch'])
                result['train_emitters_checked'] += 1
        result['train_pulses_checked'] += len(data)
        if (n + 1) % 10 == 0:
            print(f'CACHE_PROGRESS {n + 1}/250 errors={len(result["errors"])} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    catalog = raw / 'stare/tsrd_stare_train_catalog.csv'
    with catalog.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result['catalog'] = {'rows': len(rows), 'empty_configs': sum(r['is_empty'].lower() == 'true' for r in rows)}
    result['elapsed_s'] = round(time.monotonic() - started, 2)
    # This is an audit output, never an edit to any source data or configuration.
    output = Path(__file__).with_name('train250-data-verification.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    compact = {k: v for k, v in result.items() if k not in ('raw_file_hashes', 'cache_file_hashes', 'split_ids')}
    print('FINAL ' + json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
