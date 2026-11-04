#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data_pipeline_torch import TRAIN_DIR, TEST_DIR

EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
CLASS_SPECS = {
    'glioma': ('GLI',10),
    'meningioma': ('MEN',10),
    'notumor': ('NOR',10),
    'pituitary': ('PIT',10),
}
INTERNAL_CLASSES = set(CLASS_SPECS)

parser = argparse.ArgumentParser()
parser.add_argument('--web-root', type=Path, default=PROJECT_ROOT/'external_data'/'web_stress_40_raw')
parser.add_argument('--bdneuro-clean', type=Path, default=PROJECT_ROOT/'external_data'/'BDNeuro_V8_exact_clean')
parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'phase2'/'external_validation'/'web_stress_40_audit')
parser.add_argument('--near-threshold', type=int, default=6)
parser.add_argument('--strong-threshold', type=int, default=4)
args = parser.parse_args()


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in EXTS


def rel(p: Path):
    try: return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError: return str(p.resolve())


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(p: Path):
    with Image.open(p) as im:
        rgb = im.convert('RGB')
        w,h = rgb.size
        arr = np.asarray(rgb, dtype=np.uint8)
    hh = hashlib.sha256()
    hh.update(f'{w}x{h}|RGB|'.encode())
    hh.update(arr.tobytes())
    return hh.hexdigest(), w, h


def phash64(p: Path):
    with Image.open(p) as im:
        g = im.convert('L').resize((32,32), Image.Resampling.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
    t = dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')[:8,:8]
    med = np.median(t)
    v = 0
    for bit in (t >= med).reshape(-1):
        v = (v << 1) | int(bit)
    return int(v)


def add_fingerprints(records, label):
    for i,r in enumerate(records,1):
        r['sha256'] = sha256_file(r['path'])
        r['pixel_sha256'], r['width'], r['height'] = pixel_hash(r['path'])
        r['phash64'] = phash64(r['path'])
        if i % 500 == 0 or i == len(records): print(f'  {label}: {i}/{len(records)}')


def collect_web(root: Path):
    if not root.is_dir(): raise FileNotFoundError(root)
    out=[]; problems=[]
    for cls,(prefix,nexp) in CLASS_SPECS.items():
        d=root/cls
        if not d.is_dir(): problems.append(f'Missing {d}'); continue
        files=sorted(p for p in d.rglob('*') if is_image(p))
        if len(files)!=nexp: problems.append(f'{cls}: expected {nexp}, found {len(files)}')
        ids=set()
        for p in files:
            stem=p.stem.upper()
            if not stem.startswith(prefix+'_'):
                problems.append(f'{p.name}: expected {prefix}_XX')
            else:
                s=stem[len(prefix)+1:]
                if s.isdigit(): ids.add(int(s))
                else: problems.append(f'{p.name}: invalid numeric suffix')
            out.append({'dataset':'web','class':cls,'image_id':stem,'path':p})
        if ids != set(range(1,nexp+1)):
            problems.append(f'{cls}: expected IDs 01..{nexp:02d}, got {sorted(ids)}')
    if problems: raise RuntimeError('Web cohort validation failed:\n- ' + '\n- '.join(problems))
    if len(out)!=40: raise RuntimeError(f'Expected 40 images, found {len(out)}')
    return out


def collect_nick():
    out=[]
    for split,root in [('Training',Path(TRAIN_DIR)),('Testing',Path(TEST_DIR))]:
        for cd in sorted(p for p in root.iterdir() if p.is_dir()):
            cls=cd.name.lower()
            if cls not in INTERNAL_CLASSES: continue
            for p in sorted(cd.rglob('*')):
                if is_image(p): out.append({'dataset':'nickparvar','split':split,'class':cls,'path':p})
    if len(out)!=7023: raise RuntimeError(f'Expected 7023 Nickparvar images, found {len(out)}')
    return out


def collect_bd(root: Path):
    out=[]
    for cls in CLASS_SPECS:
        d=root/cls
        if not d.is_dir(): raise FileNotFoundError(d)
        for p in sorted(d.rglob('*')):
            if is_image(p): out.append({'dataset':'bdneuro_exact_clean','split':'merged','class':cls,'path':p})
    if len(out)!=2644: raise RuntimeError(f'Expected 2644 BDNeuro clean images, found {len(out)}')
    return out


def within_pairs(records,key):
    groups=defaultdict(list)
    for r in records: groups[r[key]].append(r)
    rows=[]
    for hv,g in groups.items():
        if len(g)<2: continue
        for i in range(len(g)):
            for j in range(i+1,len(g)):
                a,b=g[i],g[j]
                rows.append({'image_id_a':a['image_id'],'class_a':a['class'],'path_a':rel(a['path']),
                             'image_id_b':b['image_id'],'class_b':b['class'],'path_b':rel(b['path']),
                             'same_class':int(a['class']==b['class']),'match_hash':str(hv)})
    return rows


def cross_exact(web, ref, key, ref_name):
    lookup=defaultdict(list)
    for r in ref: lookup[r[key]].append(r)
    rows=[]
    for w in web:
        for r in lookup.get(w[key],[]):
            rows.append({'web_image_id':w['image_id'],'web_class':w['class'],'web_path':rel(w['path']),
                         'reference_dataset':ref_name,'reference_class':r['class'],'reference_path':rel(r['path']),
                         'same_class':int(w['class']==r['class']),'match_hash':str(w[key])})
    return rows


def ham(a,b): return (int(a)^int(b)).bit_count()


def cross_phash(web, ref, ref_name, near, strong):
    rows=[]
    for w in web:
        for r in ref:
            d=ham(w['phash64'],r['phash64'])
            if d<=near:
                rows.append({'phash_distance':d,'candidate_level':'STRONG_CANDIDATE' if d<=strong else 'REVIEW_CANDIDATE',
                             'web_image_id':w['image_id'],'web_class':w['class'],'web_path':rel(w['path']),
                             'reference_dataset':ref_name,'reference_class':r['class'],'reference_path':rel(r['path']),
                             'same_class':int(w['class']==r['class']),
                             'exact_file_match':int(w['sha256']==r['sha256']),
                             'exact_pixel_match':int(w['pixel_sha256']==r['pixel_sha256'])})
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    args.output_dir.mkdir(parents=True,exist_ok=True)
    print('='*80); print('WEB STRESS TEST 40 — PRE-INFERENCE AUDIT'); print('='*80)
    web=collect_web(args.web_root); nick=collect_nick(); bd=collect_bd(args.bdneuro_clean)
    print('\nCollected:'); print('Web stress-test images :',len(web)); print('Nickparvar images      :',len(nick)); print('BDNeuro clean images   :',len(bd))
    print('\nComputing fingerprints...')
    add_fingerprints(web,'Web'); add_fingerprints(nick,'Nickparvar'); add_fingerprints(bd,'BDNeuro')

    manifest=[{'image_id':r['image_id'],'class':r['class'],'path':rel(r['path']),'width':r['width'],'height':r['height'],
               'sha256':r['sha256'],'pixel_sha256':r['pixel_sha256'],'phash64_hex':f"{r['phash64']:016x}"} for r in web]
    write_csv(args.output_dir/'web_40_fingerprint_manifest.csv',manifest,
              ['image_id','class','path','width','height','sha256','pixel_sha256','phash64_hex'])

    wf=within_pairs(web,'sha256'); wp=within_pairs(web,'pixel_sha256')
    write_csv(args.output_dir/'within_web_exact_file_duplicates.csv',wf,
              ['image_id_a','class_a','path_a','image_id_b','class_b','path_b','same_class','match_hash'])
    write_csv(args.output_dir/'within_web_exact_pixel_duplicates.csv',wp,
              ['image_id_a','class_a','path_a','image_id_b','class_b','path_b','same_class','match_hash'])

    nf=cross_exact(web,nick,'sha256','nickparvar'); npix=cross_exact(web,nick,'pixel_sha256','nickparvar')
    bf=cross_exact(web,bd,'sha256','bdneuro_exact_clean'); bpix=cross_exact(web,bd,'pixel_sha256','bdneuro_exact_clean')
    write_csv(args.output_dir/'cross_exact_file_duplicates.csv',nf+bf,
              ['web_image_id','web_class','web_path','reference_dataset','reference_class','reference_path','same_class','match_hash'])
    write_csv(args.output_dir/'cross_exact_pixel_duplicates.csv',npix+bpix,
              ['web_image_id','web_class','web_path','reference_dataset','reference_class','reference_path','same_class','match_hash'])

    print('\npHash screening vs Nickparvar...')
    nph=cross_phash(web,nick,'nickparvar',args.near_threshold,args.strong_threshold)
    print('pHash screening vs BDNeuro clean...')
    bph=cross_phash(web,bd,'bdneuro_exact_clean',args.near_threshold,args.strong_threshold)
    ph=sorted(nph+bph,key=lambda r:(r['phash_distance'],r['web_image_id'],r['reference_dataset'],r['reference_path']))
    write_csv(args.output_dir/'cross_phash_candidates.csv',ph,
              ['phash_distance','candidate_level','web_image_id','web_class','web_path','reference_dataset','reference_class','reference_path','same_class','exact_file_match','exact_pixel_match'])

    best={}
    for r in ph:
        k=r['web_image_id']
        if k not in best or r['phash_distance']<best[k]['phash_distance']: best[k]=r
    best_rows=sorted(best.values(),key=lambda r:(r['phash_distance'],r['web_image_id']))
    write_csv(args.output_dir/'cross_phash_best_match_per_web_image.csv',best_rows,
              ['phash_distance','candidate_level','web_image_id','web_class','web_path','reference_dataset','reference_class','reference_path','same_class','exact_file_match','exact_pixel_match'])

    exact_ids={r['web_image_id'] for r in npix+bpix}
    ph_ids={r['web_image_id'] for r in ph}
    strong_ids={r['web_image_id'] for r in ph if r['phash_distance']<=args.strong_threshold}
    dist=Counter(r['phash_distance'] for r in best_rows)
    summary={
        'web_images_total':40,
        'web_class_counts':dict(sorted(Counter(r['class'] for r in web).items())),
        'nickparvar_reference_images':7023,
        'bdneuro_exact_clean_reference_images':2644,
        'within_web_exact_file_duplicate_pairs':len(wf),
        'within_web_exact_pixel_duplicate_pairs':len(wp),
        'cross_exact_file_duplicate_pairs':len(nf)+len(bf),
        'cross_exact_pixel_duplicate_pairs':len(npix)+len(bpix),
        'unique_web_images_with_exact_pixel_overlap':len(exact_ids),
        'unique_web_images_with_phash_le_6':len(ph_ids),
        'unique_web_images_with_phash_le_4':len(strong_ids),
        'best_match_distance_distribution':{str(k):v for k,v in sorted(dist.items())},
        'decision_rule':'Exact decoded-pixel overlap is exclusion evidence. pHash similarity alone is a review signal, not automatic exclusion proof.'
    }
    with (args.output_dir/'summary.json').open('w',encoding='utf-8') as f: json.dump(summary,f,indent=2)

    print('\n'+'='*80); print('AUDIT COMPLETE'); print('='*80)
    print('\nWITHIN WEB SET:'); print('Exact file duplicate pairs  :',len(wf)); print('Exact pixel duplicate pairs :',len(wp))
    print('\nCROSS-DATASET:'); print('Exact file duplicate pairs  :',len(nf)+len(bf)); print('Exact pixel duplicate pairs :',len(npix)+len(bpix)); print('Unique web images with exact pixel overlap:',len(exact_ids)); print(f'Unique web images with pHash <= {args.near_threshold}:',len(ph_ids)); print(f'Unique web images with pHash <= {args.strong_threshold}:',len(strong_ids))
    print('\nBEST-MATCH pHash DISTANCES')
    if dist:
        for d,n in sorted(dist.items()): print(f'distance {d}: {n} web images')
    else: print('No pHash candidates <= threshold.')
    print('\nSaved:',args.output_dir/'summary.json')

if __name__=='__main__':
    main()
