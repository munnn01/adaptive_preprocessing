import json
from pathlib import Path

def main():
    path = Path('outputs/kaggle_results/v34-od-eval-coco/outputs/v34-od-eval-coco/test_summary.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    
    print("="*80)
    print("V34 OBJECT DETECTION BENCHMARK RESULTS (COCO 2017)")
    print("Evaluator: Faster R-CNN ResNet-50 FPN")
    print("="*80)
    
    for codec in data['codecs']:
        print(f"\n--- CODEC: {codec.upper()} ---")
        curves = data['results'][codec]['curves']
        comp = data['results'][codec]['comparisons']
        
        print("\n[BD-rate comparisons relative to Anchor]")
        print(f"{'Arm':22s} | {'Cubic BD-rate':14s} | {'PCHIP BD-rate':14s} | {'BD-quality (pp)':15s}")
        print("-" * 72)
        for arm in sorted(comp.keys()):
            c = comp[arm]
            bd = f"{c['bd_rate_pct']:.2f}%" if c.get('bd_rate_pct') is not None else "N/A"
            pchip = f"{c['pchip_bd_rate_pct']:.2f}%" if c.get('pchip_bd_rate_pct') is not None else "N/A"
            qual = f"{c['bd_quality_pp']*100:+.2f}" if c.get('bd_quality_pp') is not None else "N/A"
            print(f"{arm:22s} | {bd:14s} | {pchip:14s} | {qual:15s}")
            
        print("\n[Per-QP rate (bpp) and mAP@[.50:.95]]")
        print(f"{'Arm':22s} | " + " | ".join([f"QP {q} (bpp/mAP)" for q in data['qps']]))
        print("-" * 95)
        for arm in sorted(curves.keys()):
            pts = []
            for i in range(len(data['qps'])):
                b = curves[arm]['bpp'][i]
                m = curves[arm]['quality'][i]
                pts.append(f"{b:.3f}/{m:.3f}")
            print(f"{arm:22s} | " + " | ".join(pts))

if __name__ == '__main__':
    main()
