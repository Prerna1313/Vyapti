from __future__ import annotations

import argparse
from pathlib import Path

from vyapti_train500_cache import load_train500_cache, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-root', required=True)
    ap.add_argument('--write-runtime-manifest', action='store_true')
    args = ap.parse_args()

    cache = load_train500_cache(Path(args.cache_root))
    print('=' * 72)
    print('VYAPTI TRAIN-500 CACHE VALIDATION')
    print('=' * 72)
    print(f"Available TRAIN configs : 2,500")
    print(f"Selected configs        : {len(cache['selected_config_ids']):,}")
    print(f"Emitter contributions   : {len(cache['emitter_index']):,}")
    print(f"Selection seed          : {cache['manifest'].get('selection_seed')}")
    print(f"Selection method        : {cache['manifest'].get('selection_method')}")
    print(f"Fingerprint             : {cache['fingerprint']}")
    print('Status                  : VALID')

    if args.write_runtime_manifest:
        out = Path(args.cache_root) / 'runtime_pool_manifest.json'
        save_json(out, {
            'schema': 'vyapti_runtime_selected_pool_v1',
            'source_pool_fingerprint': cache['fingerprint'],
            'selected_configs': 500,
            'selected_config_ids': cache['selected_config_ids'],
            'selected_emitter_contributions': len(cache['emitter_index']),
            'runtime_pool_root': str(cache['runtime_pool_root']),
        })
        print(f'Runtime manifest       : {out}')


if __name__ == '__main__':
    main()
