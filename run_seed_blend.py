import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

seeds = [42, 2024, 777]
output_dir = Path('output')
base_dir = Path('.')

print("=" * 60)
print("STARTING SEED BLENDING EXECUTION")
print("=" * 60)

for s in seeds:
    print(f"\\n>>> Running pipeline with SEED = {s} ...")
    # Run the pipeline
    result = subprocess.run(['python', 'nfl_draft_pipeline.py', str(s)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in pipeline with seed {s}:")
        print(result.stderr)
    else:
        print(f"Successfully finished seed {s}.")

print("\\n" + "=" * 60)
print("BLENDING PREDICTIONS")
print("=" * 60)

preds = []
for s in seeds:
    file_path = output_dir / f'submission_{s}.csv'
    if file_path.exists():
        df = pd.read_csv(file_path)
        preds.append(df['Drafted'].values)
    else:
        print(f"Warning: {file_path} not found!")

if len(preds) > 0:
    # Average the predictions
    avg_preds = np.mean(preds, axis=0)
    
    # Create final submission
    sample = pd.read_csv('input/sample_submission.csv')
    sample['Drafted'] = avg_preds
    
    final_path = output_dir / 'final_submission_blended.csv'
    sample.to_csv(final_path, index=False)
    print(f"\\nSUCCESS! Blended submission saved to: {final_path}")
    print("Averaged predictions from", len(preds), "seeds.")
else:
    print("\\nFAILED: No submission files found to blend.")
