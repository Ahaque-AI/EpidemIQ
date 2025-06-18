from neo4j import GraphDatabase
import pandas as pd
import os
from web_app.data_analysis import *

# ------------------------ Neo4j Setup ---------------------- #
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def load_existing_data():
    """Extract all case data from Neo4j into a DataFrame."""
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    query = """
    MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease)
    MATCH (c)-[:OCCURRED_IN]->(r:Region)
    MATCH (c)-[:HAS_SEVERITY]->(s:SeverityLevel)
    MATCH (p:Patient)-[:REPORTED]->(c)
    MATCH (c)-[:AFFECTED_BY]->(i:Intervention)
    OPTIONAL MATCH (c)-[:PRESENTED_SYMPTOM]->(sym:Symptom)
    
    RETURN 
        c.caseId as case_id,
        c.timestamp as timestamp,
        d.name as disease,
        r.name as region,
        s.level as severity,
        p.age as age,
        p.gender as gender,
        p.ageGroup as age_group,
        c.isOutbreakRelated as is_outbreak_related,
        c.contactTracingNeeded as contact_tracing_needed,
        c.hospitalizationRequired as hospitalization_required,
        c.location.latitude as latitude,
        c.location.longitude as longitude,
        i.type as intervention_type,
        i.effectivenessScore as intervention_effectiveness,
        i.cost as intervention_cost,
        i.complianceRate as compliance_rate,
        collect(DISTINCT sym.name) as symptoms
    """

    with driver.session() as session:
        result = session.run(query)
        records = [record.data() for record in result]

    df = pd.DataFrame(records)

    if not df.empty:
        # Convert timestamp to datetime and remove timezone
        def convert_timestamp(x):
            if hasattr(x, 'to_native'):
                dt = x.to_native()
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                return x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x

        df['timestamp'] = df['timestamp'].apply(convert_timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

df = load_existing_data()

print('Analysis of Demographic risk factors:')
print(analyze_demographic_risk_factors(df))

print('Analysis of outbreak patterns:')
print(analyze_outbreak_patterns(df))

print('Analysis of intervention effectiveness:')
print(analyze_intervention_effectiveness(df))

print('Analysis of geographic spread patterns:')
print(analyze_geographic_spread_patterns(df))

print('Analysis of temporal disease evolution:')
print(analyze_temporal_disease_evolution(df))

