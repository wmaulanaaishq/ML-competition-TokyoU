import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

seeds = [42, 2024, 777]
output_dir = Path('output')

print("=" * 60)
print("SEED BLENDING — PIPELINE V2")
print("=" * 60)

for s in seeds:
    print(f"\n>>> Running Pipeline V2 with SEED = {s} ...")
    result = subprocess.run(
        ['python', 'nfl_pipeline_v2.py', str(s)],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    if result.returncode != 0:
        # Check if it's just the cosmetic Unicode error at the end
        if 'submission_v2' in result.stdout or 'PIPELINE V2 COMPLETE' in result.stdout:
            print(f"  Seed {s} completed (cosmetic error ignored).")
        else:
            print(f"  Error with seed {s}:")
            print(result.stderr[-500:])
    else:
        print(f"  Seed {s} completed successfully.")

print("\n" + "=" * 60)
print("BLENDING PREDICTIONS")
print("=" * 60)

preds = []
for s in seeds:
    fp = output_dir / f'submission_v2_{s}.csv'
    if fp.exists():
        df = pd.read_csv(fp)
        preds.append(df['Drafted'].values)
        print(f"  Loaded {fp.name} ({len(df)} rows)")
    else:
        print(f"  WARNING: {fp} not found!")

if len(preds) >= 2:
    avg = np.mean(preds, axis=0)
    sample = pd.read_csv('input/sample_submission.csv')
    sample['Drafted'] = avg

    final_path = output_dir / 'final_submission_v2_blended.csv'
    sample.to_csv(final_path, index=False)
    print(f"\nSUCCESS! Blended {len(preds)} seeds -> {final_path}")
    print(f"Range: [{avg.min():.6f}, {avg.max():.6f}]")
else:
    print("\nFAILED: Not enough submissions to blend.")
