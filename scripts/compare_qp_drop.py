import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Add vcm_up1 to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_vcm.v33ap.evaluate import safe_curve
from adaptive_vcm.metrics import curve_summary
from adaptive_vcm.legacy_bd import bd_rate, bd_metric, _prep, _safe_polyfit
import numpy as np

def main():
    path = Path('vcm_up1/outputs/kaggle_results/v34-od-eval-coco/outputs/v34-od-eval-coco/test_summary.json')
    if not path.exists():
        path = Path('outputs/kaggle_results/v34-od-eval-coco/outputs/v34-od-eval-coco/test_summary.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    
    print("="*95)
    print("PHÂN TÍCH CHI TIẾT: BỎ QP 50 TRÊN TẬP TEST (COCO 2017 - 64 images, Faster R-CNN)")
    print("="*95)
    
    for codec in ['h264', 'h265']:
        print(f"\n############################# CODEC: {codec.upper()} #############################")
        curves = data['results'][codec]['curves']
        anchor = curves['anchor']
        
        print("\n--- BẢNG 1: TỶ LỆ TIẾT KIỆM BITRATE VÀ CHÊNH LỆCH mAP TẠI TỪNG QP ---")
        header = f"{'Arm':22s} | " + " | ".join([f"QP {q} (Δrate / ΔmAP)" for q in data['qps']])
        print(header)
        print("-" * len(header))
        for arm in ['fixed', 'forced_303401', 'forced_303402', 'forced_303403', 'forced_frozen_303401']:
            pts = []
            for i, q in enumerate(data['qps']):
                ra, qa = anchor['bpp'][i], anchor['quality'][i]
                rb, qb = curves[arm]['bpp'][i], curves[arm]['quality'][i]
                d_rate = (rb / ra - 1) * 100
                d_map = (qb - qa) * 100
                pts.append(f"{d_rate:+5.1f}% / {d_map:+5.2f}pp")
            print(f"{arm:22s} | " + " | ".join(pts))

        print("\n--- BẢNG 2: SO SÁNH BD-RATE & BD-QUALITY: 5 QPs [30-50] vs 4 QPs [30-45] ---")
        cols = f"{'Arm':22s} | {'Grid':10s} | {'Cubic BD-rate':14s} | {'PCHIP BD-rate':14s} | {'BD-quality':12s} | {'Qual Overlap [lo, hi]':22s}"
        print(cols)
        print("-" * len(cols))
        
        for arm in ['forced_303401', 'forced_303402', 'forced_303403', 'fixed', 'forced_frozen_303401']:
            for n_pts, grid_label in [(5, "5 QPs (30-50)"), (4, "4 QPs (30-45)")]:
                ra = anchor['bpp'][:n_pts]
                qa = anchor['quality'][:n_pts]
                rb = curves[arm]['bpp'][:n_pts]
                qb = curves[arm]['quality'][:n_pts]
                
                res = curve_summary(ra, qa, rb, qb)
                lo = max(min(qa), min(qb))
                hi = min(max(qa), max(qb))
                
                cb = f"{res['bd_rate_pct']:.2f}%" if res.get('bd_rate_pct') is not None else "N/A"
                pb = f"{res['pchip_bd_rate_pct']:.2f}%" if res.get('pchip_bd_rate_pct') is not None else "N/A"
                bq = f"{res['bd_quality_pp']:+.3f} pp" if res.get('bd_quality_pp') is not None else "N/A"
                ov = f"[{lo:.4f}, {hi:.4f}]"
                
                print(f"{arm:22s} | {grid_label:10s} | {cb:14s} | {pb:14s} | {bq:12s} | {ov:22s}")
            print("-" * len(cols))

if __name__ == '__main__':
    main()
