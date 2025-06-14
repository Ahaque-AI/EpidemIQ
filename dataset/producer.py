import os
import asyncio
import json
from datetime import datetime
from synthetic_dataset_creation import SyntheticHealthDataGenerator

OUTPUT_FILE = "health_records.json"

disease_category_map = {
    'gastroenteritis': 'Gastrointestinal',
    'malaria': 'Parasitic',
    'covid19': 'Viral',
    'food_poisoning': 'Bacterial/Toxin-related',
    'influenza': 'Viral',
    'meningitis': 'Infectious',
    'pneumonia': 'Respiratory'
}

async def write_json_record(record: dict):
    # Append to JSON array (or create if not exists)
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f:
            json.dump([record], f, default=str, indent=2)
    else:
        with open(OUTPUT_FILE, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

            data.append(record)
            f.seek(0)
            json.dump(data, f, default=str, indent=2)

async def produce_rows_every_second():
    generator = SyntheticHealthDataGenerator(seed=42)

    try:
        while True:
            for _ in range(1):  
                health_df = generator.generate_health_reports(
                    start_date=datetime(2020, 1, 1),
                    end_date=datetime(2025, 6, 1),
                    num_reports=1,
                    balanced=False
                )
                intervention_df = generator.generate_interventions_for_cases(health_df)

                timestamp_suffix = str(int(datetime.now().timestamp()))
                health_df['case_id'] = [f"case_{timestamp_suffix}" for i in range(len(health_df))]
                health_df['patient_id'] = [f"patient_{timestamp_suffix}" for i in range(len(health_df))]
                health_df['disease_category'] = health_df['actual_disease'].map(
                    lambda d: disease_category_map.get(d, 'Uncategorized')
                )

                hr = health_df.iloc[0].to_dict()
                iv = intervention_df.iloc[0].to_dict()

                rec = {**hr, **{f"intervention_{k}": v for k, v in iv.items()}}
                await write_json_record(rec)

            print("✅ Appended 1 row to JSON")
            await asyncio.sleep(3)

    except KeyboardInterrupt:
        print("👋 Gracefully stopped.")

if __name__ == "__main__":
    asyncio.run(produce_rows_every_second())
