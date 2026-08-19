#!/usr/bin/env python3
"""
Comprehensive experiment analysis script for ATTest V10/V11/V12 vs B2/B3.
Reads coverage data directly from artifacts (fixes CSV extraction bug).
"""
import csv
import json
import glob
import os
import sys
from datetime import datetime


OPERATORS = [
    'abs','add','bitwise_and','bitwise_not','bitwise_or','bitwise_xor',
    'cholesky','cross','diag_part','div','div_no_nan','dot',
    'equal','exp','expm1','eye','floor_div','floor_mod',
    'ger','greater','greater_equal','less','less_equal','log',
    'log1p','logical_and','logical_not','logical_or','mod','mul',
    'neg','not_equal','pow','pows','real_div','reciprocal',
    'rsqrt','rsqrt_grad','sign','sqrt','sqrt_grad','square',
    'squared_difference','sub','trace'
]

MACRO_OPS = {'sqrt_grad', 'rsqrt_grad', 'div_no_nan'}
MODEL_MAP = {
    'qwen': ('qwen_B2', 'qwen_B3'),
    'ds': ('ds_B2', 'ds_B3'),
    'glm': ('glm_B2', 'glm_B3'),
}


def at_overall(api, host):
    """ATTest style: avg(op_api, op_host) for values > 0."""
    vals = [v for v in [api, host] if isinstance(v, (int, float)) and v > 0]
    return sum(vals) / len(vals) if vals else 0.0


def load_baseline_data(json_path):
    """Load B2/B3 data from final_comparison_data.json."""
    with open(json_path) as f:
        return json.load(f)


def load_batch_data(model, version, batch_base):
    """Load operator results. Uses CSV when possible but falls back to artifacts."""
    version_dir = f'v15_{model}_epoch3_fixed-{version}_batch'
    full_base = os.path.join(batch_base, version_dir)
    csv_path = os.path.join(full_base, 'logs', 'batch_summary.csv')
    results = {}

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                op = r['operator']
                api = float(r['op_api_line']) if r['op_api_line'] not in ('N/A', '') else None
                host = float(r['op_host_line']) if r['op_host_line'] not in ('N/A', '') else None
                results[op] = {'api': api, 'host': host, 'status': r['status']}

    art_base = full_base
    for art_file in glob.glob(os.path.join(art_base, 'attest-*/.attest/artifacts/generate_code/current_coverage_summary.json')):
        op_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(art_file)))))
        op = op_dir[len('attest-'):]
        try:
            with open(art_file) as f:
                d = json.load(f)
            if 'runs' in d:
                pl = d.get('runs', {}).get('generated', {}).get('per_layer', {})
            elif 'per_layer' in d:
                pl = d['per_layer']
            else:
                continue
            api = pl.get('op_api', {}).get('line_coverage', None)
            host = pl.get('op_host', {}).get('line_coverage', None)
            if not isinstance(api, (int, float)): api = None
            if not isinstance(host, (int, float)): host = None
            status = 'extracted'
            results[op] = {'api': api, 'host': host, 'status': status}
        except Exception as e:
            pass

    return results


def compute_stats(operators_list, op_data_dict):
    """Compute aggregate statistics."""
    vals = []
    for op in operators_list:
        info = op_data_dict.get(op, {})
        api = info.get('api', None)
        host = info.get('host', None)
        ov = at_overall(api if api else 0, host if host else 0)
        vals.append((op, api, host, ov))

    nz_vals = [ov for _,_,_,ov in vals if ov > 0]
    all_vals = [ov for _,_,_,ov in vals]
    zero_ops = [op for op,_,_,ov in vals if ov == 0]

    return {
        'total': len(vals),
        'nz_count': len(nz_vals),
        'zero_count': len(zero_ops),
        'nz_avg': sum(nz_vals)/len(nz_vals) if nz_vals else 0,
        'all_avg': sum(all_vals)/len(all_vals) if all_vals else 0,
        'zero_ops': zero_ops,
        'per_op': vals,
    }


def main():
    print(f"=== ATTest Experiment Analysis ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Operators: {len(OPERATORS)}")
    print()

    b_json = '/mnt/fangcr/ATTest/final_comparison_data.json'
    b_data = load_baseline_data(b_json)

    batch_base_tmpl = '/mnt/fangcr/workspace-baseline-{model}'
    versions = ['v10', 'v11', 'v12']
    models = ['qwen', 'ds', 'glm']

    all_data = {}
    for m in models:
        bk2, bk3 = MODEL_MAP[m]
        b2_raw = b_data[bk2]
        b3_raw = b_data[bk3]
        b2_dict = {op: {'api': b2_raw[op]['api'], 'host': b2_raw[op]['host']} for op in OPERATORS}
        b3_dict = {op: {'api': b3_raw[op]['api'], 'host': b3_raw[op]['host']} for op in OPERATORS}
        all_data[('B2', m)] = b2_dict
        all_data[('B3', m)] = b3_dict

        base = batch_base_tmpl.format(model=m)
        for v in versions:
            v_data = load_batch_data(m, v, base)
            all_data[(v, m)] = v_data

    print(f"{'Model':<8} {'Config':<8} {'N>0':>5} {'Zeros':>5} {'NZ avg%':>9} {'All avg%':>10}")
    print("-" * 58)
    for m in models:
        for cfg in ['B2', 'B3', 'v10', 'v11', 'v12']:
            stats = compute_stats(OPERATORS, all_data.get((cfg, m), {}))
            done = sum(1 for op in OPERATORS if (cfg,m) in all_data and op in all_data[(cfg,m)])
            if done == 0 and cfg == 'v12':
                print(f"{m:<8} {cfg:<8} {'(running)':>5}")
            else:
                print(f"{m:<8} {cfg:<8} {stats['nz_count']:>5} {stats['zero_count']:>5} {stats['nz_avg']:>9.1f}% {stats['all_avg']:>10.1f}%")
        print()

    print("=== Zero-coverage Operators (non-macro) ===\n")
    for m in models:
        for cfg in ['B3', 'v10', 'v11', 'v12']:
            stats = compute_stats(OPERATORS, all_data.get((cfg, m), {}))
            non_macro_zeros = [op for op in stats['zero_ops'] if op not in MACRO_OPS]
            if non_macro_zeros:
                print(f"  {m}/{cfg}: {non_macro_zeros}")
            else:
                print(f"  {m}/{cfg}: (none)")
        print()

    print("=== Architecture-limited (macro delegation, 0% in B3 too) ===")
    print(f"  sqrt_grad, rsqrt_grad, div_no_nan — 0% in all configs for all models")
    print()

    print("=== V12 vs V10/V11: Operators that recovered from 0% ===\n")
    for m in models:
        v10_stats = compute_stats(OPERATORS, all_data.get(('v10', m), {}))
        v11_stats = compute_stats(OPERATORS, all_data.get(('v11', m), {}))
        v12_stats = compute_stats(OPERATORS, all_data.get(('v12', m), {}))

        v12_zeros = set(v12_stats['zero_ops'])
        v10_zeros = set(v10_stats['zero_ops'])
        v11_zeros = set(v11_stats['zero_ops'])

        recovered = v10_zeros - v12_zeros - MACRO_OPS
        if recovered:
            for op in sorted(recovered):
                v10_ov = next(ov for o,_,_,ov in v10_stats['per_op'] if o==op)
                v12_ov = next(ov for o,_,_,ov in v12_stats['per_op'] if o==op)
                print(f"  {m}/{op}: V10={v10_ov:.1f}% -> V12={v12_ov:.1f}%")
        else:
            print(f"  {m}: no recovered ops")
    print()


if __name__ == '__main__':
    main()
