#!/usr/bin/env python3
"""Concatenate ordered MMDetection output shards into one pickle file."""

import argparse
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Merge ordered MMDetection result pickle shards.')
    parser.add_argument('--shards', nargs='+', required=True, help='Input result pickle files in dataset order.')
    parser.add_argument('--output', required=True, help='Merged output pickle file.')
    parser.add_argument('--expected-count', type=int, default=10991, help='Expected number of merged results.')
    return parser.parse_args()


def main():
    args = parse_args()
    merged = []
    for shard_name in args.shards:
        shard_path = Path(shard_name)
        with shard_path.open('rb') as handle:
            shard = pickle.load(handle)
        if not isinstance(shard, list):
            raise TypeError(f'{shard_path} does not contain a result list')
        merged.extend(shard)
    if len(merged) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} results, got {len(merged)}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Merged {len(merged)} results into {output}')


if __name__ == '__main__':
    main()
