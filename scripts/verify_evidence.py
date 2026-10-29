"""Cross-check recorded evidence. Does not rerun models or verify absent datasets."""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[1]

def check():
    load=lambda name:json.loads((ROOT/'evidence'/name).read_text(encoding='utf-8'))
    summary=load('project_summary.json');dense=load('dense_dino_audit.json');data=load('selected66_provenance_audit.json')
    for field in ['episodes','frames']:
        actual=summary['dataset'][field]
        if actual!=data[field] or actual!=dense['dataset_'+field]:raise ValueError(f'Data count mismatch: {field}')
    if summary['dense_vision']['tokens']!=math.prod(dense['dense_vision_branch']['patch_grid']):raise ValueError('Token-grid mismatch')
    if dense['verification']['real_fixed_noise_equivalence_max_abs']!=0:raise ValueError('Zero-gate historical record mismatch')
    if dense['validation_split']!=0 or dense['test_split']!=0:raise ValueError('Unexpected split record')
    if summary['performance']['hardware_success_rate'] is not None:raise ValueError('Unrecorded success rate must remain unknown')
    if summary['video']['model_variant'] is not None:raise ValueError('Unattributed video must not be assigned a model')
    print(json.dumps({'status':'PASS_RECORD_CONSISTENCY','dataset_episodes':data['episodes'],'dataset_frames':data['frames'],
        'hardware_success_rate':None,'model_test_status':'32 historical tests reported; not rerun by this script'},indent=2))

if __name__=='__main__':check()
