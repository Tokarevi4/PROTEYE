# scripts/download_more_pdbs.py

import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw_pdb")
RAW_DIR.mkdir(exist_ok=True, parents=True)

TARGET = 5000

existing = {
    p.stem.lower()
    for p in RAW_DIR.glob("*.pdb")
}

print("Уже есть:", len(existing))

query = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute":
                        "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": 2.5
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute":
                        "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "greater",
                    "value": 0
                }
            }
        ]
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 10000
        }
    }
}

url = "https://search.rcsb.org/rcsbsearch/v2/query"

r = requests.post(url, json=query)
r.raise_for_status()

results = r.json()["result_set"]

ids = [
    x["identifier"].lower()
    for x in results
]

ids = [x for x in ids if x not in existing]

print("Новых найдено:", len(ids))

ids = ids[:TARGET]

for pdb_id in tqdm(ids):
    pdb_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"

    try:
        r = requests.get(pdb_url, timeout=20)

        if r.status_code != 200:
            continue

        with open(RAW_DIR / f"{pdb_id}.pdb", "w") as f:
            f.write(r.text)

    except Exception:
        pass

print("Готово.")