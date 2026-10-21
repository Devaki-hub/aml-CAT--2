import json
import urllib.request
from pathlib import Path

import pandas as pd

csv_path = Path("data/raw/extracted/PhiUSIIL_Phishing_URL_Dataset.csv")
df = pd.read_csv(csv_path)

with urllib.request.urlopen("http://localhost:8000/features") as response:
    api_info = json.load(response)

features = api_info["required_features"]

missing = [name for name in features if name not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing features: {missing}")

out_dir = Path("outputs/test_payloads")
out_dir.mkdir(parents=True, exist_ok=True)

print("Dataset labels found:", sorted(df["label"].dropna().unique().tolist()))

for label in sorted(df["label"].dropna().unique()):
    row = df.loc[df["label"] == label, features].iloc[0]

    payload = {
        "features": {
            name: float(row[name])
            for name in features
        }
    }

    output_file = out_dir / f"sample_label_{label}.json"
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Created: {output_file}")
    print(f"Feature count: {len(payload['features'])}")

print("Done. These files contain real dataset feature values.")
