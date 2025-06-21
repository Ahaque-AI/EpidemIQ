import os
import asyncio
import json
import uuid
from datetime import datetime
from neo4j import GraphDatabase
import pandas as pd
from synthetic_dataset_creation import SyntheticHealthDataGenerator

# File paths
OUTPUT_FILE = "health_records.json"
ALL_DATA_FILE = "all_data.json"

# Map diseases to categories
DISEASE_CATEGORY_MAP = {
    'gastroenteritis': 'Gastrointestinal',
    'malaria': 'Parasitic',
    'covid19': 'Viral',
    'food_poisoning': 'Bacterial/Toxin-related',
    'influenza': 'Viral',
    'meningitis': 'Infectious',
    'pneumonia': 'Respiratory'
}

# ------------------
# JSON Writing Utils
# ------------------
def load_existing_ids() -> set:
    # Load existing case_ids from ALL_DATA_FILE to ensure uniqueness

    if not os.path.exists(ALL_DATA_FILE):
        return set()
    
    df = pd.read_json(ALL_DATA_FILE, orient='records')
    
    return set(df['case_id'])

def get_unique_case_id(existing_ids: set) -> str:
    """Generate a non-duplicate case_id."""
    while True:
        uid = uuid.uuid4().hex
        case_id = f"case_{uid}"
        if case_id not in existing_ids:
            existing_ids.add(case_id)
            return case_id

async def write_json_record(record: dict):
    """Append record to OUTPUT_FILE (as array) and ALL_DATA_FILE (as JSON Lines)."""
    # 1) OUTPUT_FILE as proper JSON array
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump([record], f, default=str, indent=2)
    else:
        with open(OUTPUT_FILE, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(record)
            f.seek(0)
            json.dump(data, f, default=str, indent=2)
            f.truncate()

    # 2) ALL_DATA_FILE as JSON Lines
    with open(ALL_DATA_FILE, 'r+', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append(record)
        f.seek(0)
        json.dump(data, f, default=str, indent=2)
        f.truncate()

# -----------------------
# Main Producer Coroutine
# -----------------------
async def produce_rows_every_second():
    generator = SyntheticHealthDataGenerator(seed=42)
    existing_ids = load_existing_ids()

    # empty out the json file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    while True:
        # Generate one health report
        health_df = generator.generate_health_reports(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2025, 6, 1),
            num_reports=1,
            balanced=False
        )
        # Generate matching intervention
        intervention_df = generator.generate_interventions_for_cases(health_df)

        # Pick a guaranteed-unique case_id and patient_id
        case_id = get_unique_case_id(existing_ids)
        patient_id = f"patient_{case_id.split('_',1)[1]}"

        # Enrich the health dataframe
        health_df['case_id'] = case_id
        health_df['patient_id'] = patient_id
        health_df['disease_category'] = health_df['actual_disease'].map(
            lambda d: DISEASE_CATEGORY_MAP.get(d, 'Uncategorized')
        )

        # Combine health & intervention into one record
        hr = health_df.iloc[0].to_dict()
        iv = intervention_df.iloc[0].to_dict()
        record = {**hr, **{f"intervention_{k}": v for k, v in iv.items()}}

        # Write to both JSON files
        await write_json_record(record)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Appended record with case_id={case_id}")

        await asyncio.sleep(3)

if __name__ == "__main__":

    # empty out the json file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing file: {OUTPUT_FILE}")

    try:
        asyncio.run(produce_rows_every_second())
    except KeyboardInterrupt:
        print("Gracefully stopped producer.")