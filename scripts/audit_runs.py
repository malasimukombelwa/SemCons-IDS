#!/usr/bin/env python3
"""Read-only inventory of SemCons-IDS source runs for CRiSIS 2026.

Does NOT import joblib or load model pickles; does not train or evaluate.
Outputs: audit_runs.csv and audit_summary.json. No claims of performance.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REQUIRED = ('run_metadata.json','label_encoder.joblib','imputer.joblib',
            'variance_filter.joblib','scaler.joblib','input_features.joblib',
            'numeric_features.joblib','text_features.joblib',
            'xgboost_model.joblib','slm_adapter/adapter_config.json',
            'slm_adapter/adapter_model.safetensors')
EXPECTED = {(p,b,s) for p in ('iid_stratified','temporal_within_file')
            for b in ('minilm','modernbert') for s in (42,123,2026)}

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('reports/crisis_audit'))
    args=parser.parse_args()
    if not args.models_root.is_dir():
        parser.error('Models root does not exist: '+str(args.models_root))
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    actual=[]
    for run in sorted(p for p in args.models_root.iterdir() if p.is_dir()):
        metadata_path=run/'run_metadata.json'
        if not metadata_path.is_file():
            continue
        try:
            meta=json.loads(metadata_path.read_text(encoding='utf-8'))
        except (ValueError, OSError) as exc:
            print(f'INVALID METADATA {run.name}: {exc}')
            continue
        protocol=str(meta.get('protocol',''))
        backbone=str(meta.get('backbone',''))
        seed=meta.get('seed')
        try: seed=int(seed)
        except (TypeError,ValueError): seed=-1
        actual.append((protocol,backbone,seed))
        missing=[n for n in REQUIRED if not (run/n).is_file() or (run/n).stat().st_size==0]
        adapter_config=run/'slm_adapter/adapter_config.json'
        conf={}
        if adapter_config.is_file():
            try: conf=json.loads(adapter_config.read_text(encoding='utf-8'))
            except (ValueError,OSError): missing.append('invalid adapter_config.json')
        rows.append({
            'run':run.name,'protocol':protocol,'backbone':backbone,'seed':seed,
            'checkpoint':str(meta.get('model_checkpoint','')),
            'adapter_base':str(conf.get('base_model_name_or_path','')),
            'lora_task':str(conf.get('task_type','')),
            'lora_r':str(conf.get('r','')),
            'lora_alpha':str(conf.get('lora_alpha','')),
            'has_saved_classifier':str('classifier' in conf.get('modules_to_save',[])),
            'adapter_size_bytes':(run/'slm_adapter/adapter_model.safetensors').stat().st_size if (run/'slm_adapter/adapter_model.safetensors').is_file() else 0,
            'metadata_sha256':sha256(metadata_path),
            'adapter_sha256':sha256(run/'slm_adapter/adapter_model.safetensors') if (run/'slm_adapter/adapter_model.safetensors').is_file() else '',
            'missing_artifacts':';'.join(missing),
            'artifacts_complete':not missing,
        })
    output=args.out/'audit_runs.csv'
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['run'])
        writer.writeheader();writer.writerows(rows)
    actual_set=set(actual)
    result={'model_root':str(args.models_root),'n_runs':len(rows),
            'expected_runs':len(EXPECTED),'complete_artifact_runs':sum(r['artifacts_complete'] for r in rows),
            'missing_run_combinations':[{'protocol':p,'backbone':b,'seed':s} for p,b,s in sorted(EXPECTED-actual_set)],
            'unexpected_run_combinations':[{'protocol':p,'backbone':b,'seed':s} for p,b,s in sorted(actual_set-EXPECTED)],
            'duplicate_combinations':[{'protocol':p,'backbone':b,'seed':s,'count':n} for (p,b,s),n in Counter(actual).items() if n>1],
            'no_predictions_checked':True,'no_performance_verified':True,
            'notes':['Inventory and cryptographic checksums only.',
                     'Need original held-out per-flow predictions and true labels for paired metrics.',
                     'Need split membership/row identifiers to verify train-test and cross-file leakage.',
                     'Need original training notebook to verify inf-to-NaN preprocessing equivalence.']}
    (args.out/'audit_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('CSV:',output)
    print('JSON:',args.out/'audit_summary.json')

if __name__=='__main__':
    main()
