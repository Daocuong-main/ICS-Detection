import os
import json
import pandas as pd

# 1) Where to start searching for JSON reports
BASE_DIR = '.'  # or './results' or wherever

records = []

for root, dirs, files in os.walk(BASE_DIR):
    for fname in files:
        # adjust this test to match your report filenames
        if fname.endswith('_report.json') or fname == 'classification_report.json':
            path = os.path.join(root, fname)
            with open(path, 'r') as f:
                report = json.load(f)

            # 2) Flatten nested metrics:
            flat = {}
            for key, val in report.items():
                if isinstance(val, dict):
                    # e.g. report['0']['precision'] → column "0_precision"
                    for metric_name, metric_value in val.items():
                        flat[f'{key}_{metric_name}'] = metric_value
                else:
                    # e.g. report['accuracy']
                    flat[key] = val

            # 3) Annotate with folder & file
            flat['folder'] = os.path.relpath(root, BASE_DIR)
            flat['file']   = fname

            records.append(flat)

# 4) Build DataFrame and write out
df = pd.DataFrame(records)

# Optionally reorder columns so folder/file come first
cols = ['folder', 'file'] + [c for c in df.columns if c not in ('folder','file')]
df = df[cols]

# 5) Save
df.to_csv('all_classification_reports_summary.csv', index=False)

print(f"Wrote summary for {len(df)} reports to all_classification_reports_summary.csv")
