import os
import json
import asyncio
from datetime import datetime
from neo4j import GraphDatabase

# Path to the JSON file produced by the producer
INPUT_FILE = "health_records.json"

# Neo4j connection parameters
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Parameterized Cypher ingestion query
INGESTION_QUERY = """
UNWIND [$record] AS row

// 1. Core entities
MERGE (patient:Patient {patientId: row.patient_id})
  ON CREATE SET
    patient.age      = toInteger(row.age),
    patient.ageGroup = row.age_group,
    patient.gender   = row.gender
MERGE (disease:Disease {name: row.actual_disease})
MERGE (case_region:Region {name: row.region})
MERGE (intervention_region:Region {name: row.intervention_region})
MERGE (target_disease:Disease {name: row.intervention_target_disease})

// 2. Conceptual nodes
MERGE (severity:SeverityLevel {level: row.severity})
MERGE (category:DiseaseCategory {name: row.disease_category})

// 3. Demographics
MERGE (ageGroupNode:AgeGroup {name: row.age_group})
MERGE (genderNode:Gender {value: row.gender})

// 4. Intervention
MERGE (intervention:Intervention {id: row.intervention_intervention_id})
  ON CREATE SET
    intervention.type               = row.intervention_type,
    intervention.startDate          = date(split(row.intervention_start_date, ' ')[0]),
    intervention.durationDays       = toInteger(row.intervention_duration_days),
    intervention.effectivenessScore = toFloat(row.intervention_effectiveness_score),
    intervention.cost               = toInteger(row.intervention_cost),
    intervention.populationAffected = toInteger(row.intervention_population_affected),
    intervention.complianceRate     = toFloat(row.intervention_compliance_rate)

// 5. Time‑tree
MERGE (year:Year {year: row.timestamp.year})
MERGE (month:Month {month: row.timestamp.month})
MERGE (day:Day {day: row.timestamp.day})
MERGE (weekday:DayOfWeek {name: row.day_of_week})
MERGE (hour:Hour {hour: row.hour})

// 6. Case node
CREATE (case:Case {
  caseId                 : row.case_id,
  timestamp              : row.timestamp,
  location               : point({latitude: toFloat(row.latitude), longitude: toFloat(row.longitude)}),
  isOutbreakRelated      : toBoolean(row.is_outbreak_related),
  contactTracingNeeded   : toBoolean(row.contact_tracing_needed),
  hospitalizationRequired: toBoolean(row.hospitalization_required)
})

// 7. Core relationships
MERGE (patient)-[:REPORTED]->(case)
MERGE (case)-[:DIAGNOSED_WITH]->(disease)
MERGE (case)-[:HAS_SEVERITY]->(severity)
MERGE (case)-[:OCCURRED_IN]->(case_region)
MERGE (case)-[:AFFECTED_BY]->(intervention)

// 8. Demographics relations
MERGE (patient)-[:BELONGS_TO_AGE_GROUP]->(ageGroupNode)
MERGE (patient)-[:HAS_GENDER]->(genderNode)

// 9. Time relations
MERGE (case)-[:OCCURRED_ON]->(day)
MERGE (day)-[:OF_MONTH]->(month)
MERGE (month)-[:OF_YEAR]->(year)
MERGE (day)-[:IS_WEEKDAY]->(weekday)
MERGE (case)-[:OCCURRED_AT_HOUR]->(hour)
MERGE (case)-[:REPORTED_IN_YEAR]->(year)

// 10. Symptoms
FOREACH (symptom_name IN row.symptoms_text |
  MERGE (symptom:Symptom {name: symptom_name})
  MERGE (case)-[:PRESENTED_SYMPTOM]->(symptom)
  MERGE (symptom)-[:COMMON_MANIFESTATION_OF]->(disease)
)

// 11. Intervention relations
MERGE (intervention)-[:TARGETS_DISEASE]->(target_disease)
MERGE (intervention)-[:APPLIED_IN]->(intervention_region)

// 12. Category & history
MERGE (disease)-[:IS_A_TYPE_OF]->(category)
MERGE (patient)-[:LIVES_IN]->(case_region)
MERGE (patient)-[:HAS_HISTORY_OF]->(disease)
MERGE (disease)-[:PREVALENT_IN]->(case_region)

// 13. Outbreak
FOREACH (_ IN CASE WHEN toBoolean(row.is_outbreak_related) THEN [1] ELSE [] END |
  MERGE (outbreak:Outbreak {disease: row.actual_disease, region: row.region})
    ON CREATE SET outbreak.startDate = row.timestamp
  MERGE (case)-[:PART_OF_OUTBREAK]->(outbreak)
)

// 14. Conditional services
FOREACH (_ IN CASE WHEN toBoolean(row.contact_tracing_needed) THEN [1] ELSE [] END |
  MERGE (ct:ContactTracingService {name: "ContactTracing"})
  MERGE (case)-[:REQUIRES_CONTACT_TRACING]->(ct)
)
FOREACH (_ IN CASE WHEN toBoolean(row.hospitalization_required) THEN [1] ELSE [] END |
  MERGE (hs:HospitalizationService {name: "Hospitalization"})
  MERGE (case)-[:REQUIRES_HOSPITALIZATION]->(hs)
)
"""

async def watch_and_ingest(file_path: str, driver):
    """
    Watches the JSON file and ingests new records in real time.
    Prints confirmation of created nodes/relationships per record.
    """
    processed = set()
    while True:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        for rec in data:
            key = rec.get('case_id') or rec.get('timestamp')
            if key in processed:
                continue

            # Parse timestamp to Python datetime
            ts = rec.get('timestamp')
            if isinstance(ts, str):
                try:
                    rec['timestamp'] = datetime.fromisoformat(ts)
                except ValueError:
                    rec['timestamp'] = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')

            # Ingest record into Neo4j
            with driver.session() as session:
                result = session.run(INGESTION_QUERY, record=rec)
                summary = result.consume()

            print(f"Ingested {key}: +{summary.counters.nodes_created} nodes, +{summary.counters.relationships_created} rels")
            processed.add(key)

        await asyncio.sleep(2)


def main():
    # Initialize Neo4j driver
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    # empty out the json file
    if os.path.exists(INPUT_FILE):
        os.remove(INPUT_FILE)
        print(f"Removed existing file: {INPUT_FILE}")

    try:
        asyncio.run(watch_and_ingest(INPUT_FILE, driver))
    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        driver.close()

if __name__ == '__main__':
    main()