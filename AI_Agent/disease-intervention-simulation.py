import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import json
from textwrap import dedent
from typing import Dict, List, Any, Optional
from scipy import stats
import re
import logging
import asyncio
import requests
import os
from fpdf import FPDF

# Add these imports at the top
import matplotlib.pyplot as plt

# Import agno agents
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.toolkit import Toolkit
from agno.tools.python import PythonTools
from agno.memory.db.postgres import PgMemoryDb
from agno.models.nvidia import Nvidia
from agno.models.google import Gemini

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class InternetSearch:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_api_key = os.getenv("SERPER_API_KEY")
        self.headers = {"Content-Type": "application/json"}

    def search(self, query: str, engine: str = "tavily", num_results: int = 5) -> Dict[str, Any]:
        if engine == "tavily":
            return self._search_tavily(query, num_results)
        elif engine == "serpapi":
            return self._search_serpapi(query, num_results)
        else:
            return {"error": f"Unknown engine: {engine}. Use 'tavily' or 'serpapi'."}

    def _search_tavily(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        if not self.tavily_api_key:
            return {"error": "Tavily API key not found in environment variables"}

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            return {"error": f"Tavily request failed: {response.text}"}
        
        data = response.json()
        results = [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in data.get("results", [])
        ]
        return {"engine": "tavily", "query": query, "results": results}

    def _search_serpapi(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        if not self.serpapi_api_key:
            return {"error": "SerpAPI key not found in environment variables"}

        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": num_results
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {"error": f"SerpAPI request failed: {response.text}"}
        
        data = response.json()
        results = [
            {"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
            for r in data.get("organic_results", [])[:num_results]
        ]
        return {"engine": "serpapi", "query": query, "results": results}

# Custom tools for epidemic data analysis
class EpidemicAnalysisTools(Toolkit):
    def __init__(self):
        super().__init__(name="epidemic_analysis_tools")
        
    def analyze_data_drift(self, df: pd.DataFrame, split_days: int = 30) -> Dict[str, Any]:
        """Analyze data drift in epidemic data"""
        if df.empty:
            return {"error": "No data available for drift analysis"}
        
        # Split data based on timestamp
        if 'timestamp' in df.columns:
            cutoff_date = df['timestamp'].max() - timedelta(days=split_days)
            historical_df = df[df['timestamp'] <= cutoff_date]
            current_df = df[df['timestamp'] > cutoff_date]
        else:
            # Fallback to index-based split
            split_point = int(len(df) * 0.7)
            historical_df = df.iloc[:split_point]
            current_df = df.iloc[split_point:]
        
        if historical_df.empty or current_df.empty:
            return {"error": "Insufficient data for drift comparison"}
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'data_periods': {
                'historical_records': len(historical_df),
                'current_records': len(current_df),
                'split_date': cutoff_date.isoformat() if 'timestamp' in df.columns else None
            },
            'drift_analysis': {},
            'summary': {}
        }
        
        # Analyze numerical columns
        numerical_cols = [col for col in ['age', 'intervention_effectiveness', 'intervention_cost', 'compliance_rate'] 
                         if col in df.columns]
        
        for col in numerical_cols:
            hist_data = historical_df[col].dropna()
            curr_data = current_df[col].dropna()
            
            if len(hist_data) > 0 and len(curr_data) > 0:
                ks_stat, p_value = stats.ks_2samp(hist_data, curr_data)
                
                drift_results['drift_analysis'][col] = {
                    'type': 'numerical',
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': p_value < 0.05,
                    'historical_mean': float(hist_data.mean()),
                    'current_mean': float(curr_data.mean()),
                    'mean_change_percent': float(((curr_data.mean() - hist_data.mean()) / hist_data.mean()) * 100)
                }
        
        # Analyze categorical columns
        categorical_cols = [col for col in ['disease', 'region', 'severity', 'gender', 'intervention_type'] 
                           if col in df.columns]
        
        for col in categorical_cols:
            hist_counts = historical_df[col].value_counts()
            curr_counts = current_df[col].value_counts()
            
            # Calculate distribution shifts
            all_categories = set(hist_counts.index) | set(curr_counts.index)
            hist_props = [(hist_counts.get(cat, 0) / len(historical_df)) for cat in all_categories]
            curr_props = [(curr_counts.get(cat, 0) / len(current_df)) for cat in all_categories]
            
            if len(all_categories) > 1:
                chi2_stat, p_value = stats.chisquare(curr_props, hist_props)
                
                drift_results['drift_analysis'][col] = {
                    'type': 'categorical',
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'drift_detected': p_value < 0.05,
                    'historical_distribution': {k: float(v/len(historical_df)) for k, v in hist_counts.items()},
                    'current_distribution': {k: float(v/len(current_df)) for k, v in curr_counts.items()}
                }
        
        # Summary
        drift_detected_count = sum(1 for analysis in drift_results['drift_analysis'].values() 
                                 if analysis.get('drift_detected', False))
        total_analyses = len(drift_results['drift_analysis'])
        
        drift_results['summary'] = {
            'total_features_analyzed': total_analyses,
            'features_with_drift': drift_detected_count,
            'drift_percentage': float(drift_detected_count / total_analyses * 100) if total_analyses > 0 else 0,
            'overall_drift_detected': drift_detected_count > 0,
            'drift_severity': 'High' if drift_detected_count > total_analyses * 0.5 else 'Medium' if drift_detected_count > 0 else 'Low'
        }
        
        return drift_results
    
    def simulate_disease_spread_from_data(self, df: pd.DataFrame, selected_disease: str, simulation_days: int = 90) -> Dict[str, Any]:
        """Simulate disease spread using patterns from actual data"""
        if df.empty:
            return {"error": "No data available for simulation"}
        
        # Filter data for selected disease
        disease_data = df[df['disease'] == selected_disease] if 'disease' in df.columns else df
        
        if disease_data.empty:
            return {"error": f"No data found for disease: {selected_disease}"}
        
        # Extract disease characteristics from data
        severity_rate = (disease_data['severity'] == 'High').mean() if 'severity' in disease_data.columns else 0.1
        hospitalization_rate = disease_data['hospitalization_required'].mean() if 'hospitalization_required' in disease_data.columns else 0.05
        
        # Calculate transmission rate from case growth
        if 'timestamp' in disease_data.columns:
            daily_cases = disease_data.groupby(disease_data['timestamp'].dt.date).size()
            if len(daily_cases) > 1:
                growth_rates = daily_cases.pct_change().dropna()
                avg_growth_rate = growth_rates.mean() if not growth_rates.empty else 0.1
            else:
                avg_growth_rate = 0.1
        else:
            avg_growth_rate = 0.1
        
        # Estimate population from data
        total_cases = len(disease_data)
        estimated_population = max(total_cases * 100, 10000)  # Assume cases are 1% of population
        
        # SEIR simulation parameters derived from data
        r0 = max(1 + avg_growth_rate * 7, 0.5)  # Convert daily growth to R0
        incubation_period = 5  # days
        infectious_period = 10  # days
        
        # Run SEIR simulation
        beta = r0 / infectious_period
        sigma = 1 / incubation_period
        gamma = 1 / infectious_period
        
        # Initialize compartments
        initial_infected = max(1, total_cases // 30)  # Estimate active cases
        S = [estimated_population - initial_infected]
        E = [0]
        I = [initial_infected]
        R = [0]
        
        # Simulate
        for day in range(1, simulation_days + 1):
            new_infections = beta * I[day-1] * S[day-1] / estimated_population
            new_symptomatic = sigma * E[day-1]
            new_recoveries = gamma * I[day-1]
            
            S.append(max(0, S[day-1] - new_infections))
            E.append(max(0, E[day-1] + new_infections - new_symptomatic))
            I.append(max(0, I[day-1] + new_symptomatic - new_recoveries))
            R.append(R[day-1] + new_recoveries)
        
        # Calculate metrics
        peak_infections = max(I)
        peak_day = I.index(peak_infections)
        total_infected = R[-1]
        
        return {
            'disease': selected_disease,
            'data_derived_parameters': {
                'r0': float(r0),
                'severity_rate': float(severity_rate),
                'hospitalization_rate': float(hospitalization_rate),
                'estimated_population': int(estimated_population),
                'historical_cases': int(total_cases)
            },
            'simulation_results': {
                'susceptible': S,
                'exposed': E,
                'infectious': I,
                'recovered': R,
                'days': list(range(simulation_days + 1))
            },
            'metrics': {
                'peak_infections': int(peak_infections),
                'peak_day': peak_day,
                'total_infected': int(total_infected),
                'attack_rate': float(total_infected / estimated_population),
                'estimated_deaths': int(total_infected * severity_rate),
                'estimated_hospitalizations': int(total_infected * hospitalization_rate)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_interventions_from_data(self, df: pd.DataFrame, selected_disease: str, baseline_simulation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate interventions based on historical effectiveness from data"""
        if df.empty:
            return {"error": "No data available for intervention analysis"}
        
        # Filter for disease-specific intervention data
        disease_data = df[df['disease'] == selected_disease] if 'disease' in df.columns else df
        
        if disease_data.empty:
            return {"error": f"No intervention data found for disease: {selected_disease}"}
        
        # Analyze interventions by type
        intervention_analysis = {}
        
        if 'intervention_type' in disease_data.columns:
            for intervention_type in disease_data['intervention_type'].unique():
                intervention_data = disease_data[disease_data['intervention_type'] == intervention_type]
                
                # Calculate metrics from actual data
                avg_effectiveness = intervention_data['intervention_effectiveness'].mean() if 'intervention_effectiveness' in intervention_data.columns else 0
                avg_cost = intervention_data['intervention_cost'].mean() if 'intervention_cost' in intervention_data.columns else 0
                avg_compliance = intervention_data['compliance_rate'].mean() if 'compliance_rate' in intervention_data.columns else 0
                
                # Calculate impact on case outcomes
                hospitalization_rate = intervention_data['hospitalization_required'].mean() if 'hospitalization_required' in intervention_data.columns else 0
                severity_rate = (intervention_data['severity'] == 'High').mean() if 'severity' in intervention_data.columns else 0
                
                # Estimate intervention impact on baseline simulation
                baseline_total = baseline_simulation.get('metrics', {}).get('total_infected', 0)
                cases_prevented = int(baseline_total * avg_effectiveness * avg_compliance)
                lives_saved = int(cases_prevented * baseline_simulation.get('data_derived_parameters', {}).get('severity_rate', 0.1))
                
                intervention_analysis[intervention_type] = {
                    'historical_cases': len(intervention_data),
                    'effectiveness_score': float(avg_effectiveness),
                    'average_cost': float(avg_cost),
                    'compliance_rate': float(avg_compliance),
                    'hospitalization_rate': float(hospitalization_rate),
                    'severity_rate': float(severity_rate),
                    'estimated_cases_prevented': cases_prevented,
                    'estimated_lives_saved': lives_saved,
                    'cost_per_case_prevented': float(avg_cost / max(cases_prevented, 1)),
                    'cost_per_life_saved': float(avg_cost / max(lives_saved, 1)),
                    'roi_score': float((avg_effectiveness * avg_compliance) / (avg_cost / 10000)) if avg_cost > 0 else 0
                }
        
        # Rank interventions
        ranked_interventions = sorted(
            intervention_analysis.items(),
            key=lambda x: x[1]['roi_score'],
            reverse=True
        )
        
        return {
            'disease': selected_disease,
            'analysis_timestamp': datetime.now().isoformat(),
            'intervention_analysis': intervention_analysis,
            'rankings': {
                'by_effectiveness': sorted(intervention_analysis.items(), 
                                         key=lambda x: x[1]['effectiveness_score'], reverse=True),
                'by_cost_effectiveness': sorted(intervention_analysis.items(), 
                                              key=lambda x: x[1]['cost_per_case_prevented']),
                'by_roi': ranked_interventions
            },
            'recommendations': {
                'most_effective': ranked_interventions[0][0] if ranked_interventions else None,
                'best_roi': ranked_interventions[0][0] if ranked_interventions else None,
                'lowest_cost': min(intervention_analysis.items(), 
                                 key=lambda x: x[1]['average_cost'])[0] if intervention_analysis else None
            }
        }

today = datetime.now().strftime("%Y-%m-%d")

# Agent 1: Data Drift Analyst
data_drift_agent = Agent(
    model=Groq('deepseek-r1-distill-llama-70b'),
    tools=[EpidemicAnalysisTools(), InternetSearch()],  # InternetSearch added
    description=dedent("""\
        You are DataDriftAnalyst, an expert epidemiological data quality analyst specializing 
        in monitoring data quality and detecting changes in disease surveillance patterns. 
        You use advanced statistical methods to identify when current data significantly 
        differs from historical patterns, ensuring model reliability for epidemic forecasting.
        
        Your expertise includes:
        - Statistical drift detection algorithms
        - Time series analysis for surveillance data
        - Data quality assessment frameworks
        - Epidemiological data validation techniques
    """),
    instructions=dedent("""\
        1. Analyze incoming surveillance data for statistical anomalies
        2. Compare current data distributions with historical baselines
        3. Apply appropriate drift detection methods (KS tests, PSI, etc.)
        4. Identify potential causes of detected drift
        5. Provide actionable recommendations for data quality improvement
        6. Generate comprehensive drift analysis reports with visualizations
        7. Flag critical drift patterns that may affect model performance
        9. Provide summary tables of drift metrics
        10. Use real-time internet search to supplement knowledge or context when needed
    """),
    expected_output=dedent("""\
    # Data Drift Analysis Report
    ## Executive Summary
    {Brief overview of drift detection results and severity}
    
    ## Data Quality Assessment
    {Current data quality metrics and comparison with historical standards}
    
    ## Drift Detection Results
    {Statistical tests results, drift magnitude, and affected variables}
    
    ## Root Cause Analysis
    {Potential causes of detected drift patterns}
    
    ## Impact Assessment
    {Effect on model reliability and surveillance accuracy}
    
    ## Recommendations
    - {Action item 1}
    - {Action item 2}
    - {Action item 3}
    
    ## Technical Details
    {Statistical methods used and detailed findings}
    
    ## Drift Metrics Table
    {Generate a table with variables, drift scores, and significance levels}
    
    ---
    Analysis by DataDriftAnalyst
    Epidemiological Data Quality Division
    Date: {current_date}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,

)

disease_simulation_agent = Agent(
    model=Groq('meta-llama/llama-4-scout-17b-16e-instruct'),
    tools=[EpidemicAnalysisTools(), InternetSearch()],
    description=dedent("""\
        You are DiseaseSpreadSimulator, a computational epidemiologist who specializes 
        in disease spread modeling and intervention scenario analysis. You create realistic simulations 
        of disease transmission using actual surveillance data to parameterize your models and predict 
        outbreak scenarios with and without public health interventions.
        
        Your expertise includes:
        - SEIR and compartmental disease models
        - Agent-based epidemic simulations
        - Parameter estimation from real-world data
        - Intervention modeling (given in task)
        - Scenario analysis with intervention strategies
        - Temporal/spatial intervention effectiveness evaluation
    """),
    instructions=dedent("""\
        1. Analyze epidemiological surveillance data to extract key parameters
        2. Run appropriate compartmental models (SEIR, SIRD, etc.) from your own knowledge
        3. Calibrate model parameters using historical outbreak data
        4. Design intervention strategies based on literature/expert consensus
        5. Simulate baseline scenario (no interventions)
        6. Simulate intervention scenarios with varying implementation timing/intensity
        7. Perform sensitivity analysis on key parameters including intervention effectiveness
        8. Generate probabilistic forecasts with confidence intervals for all scenarios
        9. Create visualizations showing intervention impact trajectories
        10. Validate model outputs against known intervention patterns
        11. Provide summary tables of key predictions and intervention metrics
        12. Format report for downstream intervention evaluation agent
        13. Use internet search to inform model parameters or improve scenario assumptions
    """),
    expected_output=dedent("""\
    # Disease Spread Simulation Report
    ## Executive Summary
    {Overview of simulation results and key predictions including intervention impact}
    
    ## Model Configuration
    - {Chosen model type}
    - {Baseline parameters and assumptions}
    - {Intervention parameters (timing, coverage, effectiveness)}
    
    ## Data Calibration
    - {How surveillance data was used to parameterize the model}
    - {Intervention calibration sources (literature/real-world data)}
    
    ## Intervention Strategies
    - {Description of simulated interventions (social distancing, vaccination, etc.)}
    - {Implementation timing and coverage assumptions}
    - {Estimated compliance levels}
    
    ## Simulation Results
    - {Baseline trajectory (no interventions)}
    - {Intervention scenario trajectories}
    - {Peak reduction metrics (cases, hospitalizations, deaths)}
    
    ## Intervention Effectiveness Analysis
    - {Comparison of scenarios by intervention type and timing}
    - {Cost-effectiveness ratios (cases averted per intervention unit)}
    - {Break-even points for intervention implementation}
    
    ## Sensitivity Analysis
    - {Impact of parameter uncertainty on predictions}
    - {Vulnerability of outcomes to intervention timing/compliance variations}
    
    ## Model Validation
    - {Comparison with historical patterns and validation metrics}
    - {Intervention validation against real-world implementation data}
    
    ## Key Predictions Table
    | Scenario          | Peak Reduction | Cases Averted | Confidence Interval | Probability |
    |-------------------|----------------|---------------|---------------------|-------------|
    | Baseline          | -              | -             | -                   | -           |
    | Early Vaccination | 45%            | 2.3M          | ±15%                | 85%         |
    | Social Distancing | 30%            | 1.2M          | ±10%                | 90%         |
    
    ## Intervention Recommendations
    - {Ranked list of most effective interventions with confidence scores}
    - {Implementation timing suggestions with probabilistic support}
    
    ## Limitations and Uncertainties
    - {Model assumptions and intervention parameter uncertainties}
    - {Data gaps in intervention effectiveness metrics}
    
    ---
    Simulation by DiseaseSpreadSimulator
    Computational Epidemiology Division
    Date: {current_date}
    Intervention Evaluation ID: {unique_id} (for downstream agent processing)
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# Agent 3: Intervention Evaluator
intervention_agent = Agent(
    model=Groq('meta-llama/llama-4-maverick-17b-128e-instruct'),
    tools=[EpidemicAnalysisTools(), InternetSearch()],  # InternetSearch added
    description=dedent("""\
        You are InterventionEvaluator, a public health policy analyst with expertise 
        in intervention effectiveness research. You analyze historical intervention data 
        to evaluate which strategies work best for different diseases and provide 
        evidence-based recommendations for public health decision-making.
        
        Your expertise includes:
        - Intervention effectiveness analysis
        - Cost-benefit assessment frameworks
        - Comparative effectiveness research
        - Policy impact evaluation
        - Evidence synthesis and meta-analysis
    """),
    instructions=dedent("""\
        1. Analyze historical intervention data across different disease outbreaks
        2. Evaluate effectiveness metrics for various intervention strategies
        3. Conduct cost-benefit analysis for each intervention type
        4. Rank interventions based on effectiveness, feasibility, and cost
        5. Consider contextual factors (population, geography, resources)
        6. Synthesize evidence from multiple sources and studies
        7. Provide clear recommendations with supporting evidence
        8. Account for implementation challenges and resource constraints
        10. Provide summary tables of intervention rankings and metrics
        11. Use internet search to gather recent evidence and effectiveness studies
    """),
    expected_output=dedent("""\
    # Intervention Effectiveness Analysis
    ## Executive Summary
    {Overview of evaluated interventions and top recommendations}
    
    ## Methodology
    {Analysis approach, data sources, and evaluation criteria}
    
    ## Intervention Assessment
    {Detailed evaluation of each intervention strategy}
    
    ## Effectiveness Rankings
    {Ranked list of interventions by effectiveness metrics}
    
    ## Cost-Benefit Analysis
    {Economic evaluation and resource requirements}
    
    ## Contextual Considerations
    {Factors affecting intervention success in different settings}
    
    ## Evidence Quality
    {Strength of evidence and reliability of findings}
    
    ## Recommendations
    - {Top recommendation with rationale}
    - {Second recommendation with rationale}
    - {Third recommendation with rationale}
    
    ## Implementation Guidance
    {Practical considerations for deployment}
    
    ## Monitoring and Evaluation
    {Metrics for tracking intervention success}
    
    ## Intervention Rankings Table
    {Generate a table with interventions, effectiveness scores, costs, and feasibility ratings}
    
    ---
    Analysis by InterventionEvaluator
    Public Health Policy Division
    Date: {current_date}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    reasoning=True
)

def encode_text(text):
    """
    Encodes a string to latin-1, replacing any characters that cannot be encoded.
    This is crucial for FPDF which does not handle UTF-8 characters natively.
    """
    replacements = {
        '\u2014': '-',  # Em dash
        '\u2013': '-',  # En dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2022': '*',  # Bullet
    }
    for unicode_char, latin_char in replacements.items():
        text = text.replace(unicode_char, latin_char)
    return text.encode('latin-1', 'replace').decode('latin-1')

# --- 2. The Final PDF Class with All Enhancements ---
class FinalReportPDF(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4', report_info=None):
        super().__init__(orientation, unit, format)
        self.report_info = report_info or {}
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=20)
        self.is_first_content_page = True

    def header(self):
        if self.page_no() == 1: return
        if self.is_first_content_page:
            self.set_y(15); self.set_font('Arial', 'B', 14); self.set_text_color(0, 51, 102)
            self.cell(0, 10, encode_text(self.report_info.get('title', 'Report')), 0, 1, 'L')
            self.set_font('Arial', '', 10); self.set_text_color(80, 80, 80)
            self.cell(0, 5, encode_text(f"Generated on: {self.report_info.get('generation_date', '')}"), 0, 1, 'L')
            self.ln(5)
            self.is_first_content_page = False
        else:
            self.set_y(10); self.set_font('Arial', 'B', 9); self.set_text_color(128, 128, 128)
            self.cell(0, 10, encode_text(self.report_info.get('title', 'Report')), 0, 1, 'L')
            self.set_y(20)

    def footer(self):
        if self.page_no() == 1: return
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no() - 1}', 0, 0, 'C')

    def create_title_page(self):
        self.add_page(); self.set_y(80); self.set_font('Arial', 'B', 28); self.set_text_color(0, 51, 102)
        self.multi_cell(0, 12, encode_text(self.report_info.get('title')), align='C')
        self.ln(20); self.set_font('Arial', '', 16); self.set_text_color(80, 80, 80)
        self.multi_cell(0, 10, encode_text(self.report_info.get('subtitle')), align='C')
        self.set_y(230); self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, encode_text(f"Analysis by: {self.report_info.get('author', '')}"), align='C')
        self.multi_cell(0, 7, encode_text(f"Date: {self.report_info.get('date', '')}"), align='C')

    # --- UPDATED AND NEW METHODS ---

    def add_section_header(self, title): # Handles # Heading 1
        self.ln(5)
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f' {encode_text(title)}', 0, 1, 'L', True)
        self.ln(4)

    def add_subsection_header(self, title): # NEW: Handles ## Heading 2
        self.ln(3)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(40, 40, 40) # Dark Gray
        self.cell(0, 8, encode_text(title), 0, 1, 'L')
        self.ln(2)

    def add_body_text(self, text): # REWRITTEN: Handles paragraphs, bullets, and **bold** text
        """
        Parses a line of text for bullets and inline bolding (**) and writes it to the PDF.
        """
        safe_text = encode_text(text.strip())
        line_height = 7 # Set consistent line height

        # Check for and handle bullet points
        is_bullet = False
        if safe_text.startswith('* ') or safe_text.startswith('- '):
            is_bullet = True
            safe_text = safe_text[2:] # Remove the marker

        if is_bullet:
            self.cell(5) # Indent for the bullet
            self.cell(5, line_height, chr(149)) # Bullet character
        
        # Split the line by the bold delimiter '**'
        parts = safe_text.split('**')
        
        for i, part in enumerate(parts):
            if not part: continue # Skip empty parts
            
            # Parts at odd indices are bold
            is_bold = (i % 2 == 1)
            
            current_style = 'B' if is_bold else ''
            self.set_font('Arial', current_style, 11)
            
            # Use write() to append text on the same line
            self.write(line_height, part)

        self.ln() # Move to the next line after the full line is written
        self.ln(3) # Add a little space after the paragraph


    def add_image_with_caption(self, path, caption, description):
        self.ln(5)
        if os.path.exists(path):
            self.image(path, w=self.w - 30)
            self.ln(2)
        else:
            self.set_font('Arial', 'BI', 10); self.set_text_color(255, 0, 0)
            self.multi_cell(0, 6, encode_text(f"[Image not found: {path}]"))
        
        # The description below the image is now handled by the powerful add_body_text
        if description:
            self.add_body_text(description)
        self.ln(5)

    def add_table(self, data_lines):
        if not data_lines: return
        rows = [re.split(r'\s*\|\s*', line.strip())[1:-1] for line in data_lines]
        if not rows: return
        header, data = rows[0], rows[1:]
        widths = [(self.w - 30) * 0.45, (self.w - 30) * 0.2, (self.w - 30) * 0.35]
        self.set_font('Arial', 'B', 10); self.set_fill_color(0, 51, 102); self.set_text_color(255, 255, 255)
        for i, h in enumerate(header):
            if i < len(widths): self.cell(widths[i], 9, encode_text(h.strip()), 1, 0, 'C', True)
        self.ln()
        self.set_font('Arial', '', 10); self.set_text_color(0, 0, 0)
        for row in data:
            self.set_fill_color(255, 255, 255)
            for i, item in enumerate(row):
                if i < len(widths): self.cell(widths[i], 9, encode_text(item.strip()), 1, 0, 'L', True)
            self.ln()
        self.ln(8)


# --- 3. The Main PDF Generation Function with All Logic ---
def create_final_report(report_markdown: str, filename: str, report_info: dict):
    try:
        # NEW: Pre-process to remove <think> tags.
        # re.DOTALL makes '.' match newlines, so multi-line tags are removed.
        report_markdown = re.sub(r'<think>.*?</think>', '', report_markdown, flags=re.DOTALL)
        
        pdf = FinalReportPDF(report_info=report_info)
        pdf.create_title_page()
        pdf.add_page()

        lines = report_markdown.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line: i += 1; continue

            # UPDATED: Check for heading levels in the correct order
            if line.startswith('### '):
                pdf.add_subsection_header(line[4:])
            elif line.startswith('## '):
                pdf.add_subsection_header(line[3:])
            elif line.startswith('# '):
                pdf.add_section_header(line[2:])
            elif (match := re.match(r'!\[(.*)\]\((.*)\)', line)):
                caption, path = match.groups()
                description = ""
                is_next_line_a_description = (
                    i + 1 < len(lines) and lines[i+1].strip() and 
                    not lines[i+1].strip().startswith(('#', '!', '|', '*', '-'))
                )
                if is_next_line_a_description:
                    description = lines[i+1].strip()
                    i += 1
                pdf.add_image_with_caption(path, caption, description)
            elif line.startswith('|'):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    if not re.match(r'^[|\s:-]+$', lines[i]):
                        table_lines.append(lines[i])
                    i += 1
                pdf.add_table(table_lines)
                i -= 1
            else:
                # All other lines (including bullets and bolded text) go here
                pdf.add_body_text(line)
            i += 1

        pdf.output(filename)
        print(f"✅ Final, feature-complete report saved as '{filename}'")

    except Exception as e:
        print(f"❌ Error generating final report: {e}")
        import traceback
        traceback.print_exc()

def get_user_disease_selection(available_diseases):
    """Interactive function to get disease selection from user"""
    print("\n" + "="*50)
    print("DISEASE SELECTION")
    print("="*50)
    print("Available diseases in the surveillance data:")
    
    for i, disease in enumerate(available_diseases, 1):
        print(f"{i}. {disease}")
    
    print(f"{len(available_diseases) + 1}. Skip disease-specific analysis")
    
    while True:
        try:
            choice = input(f"\nSelect a disease (1-{len(available_diseases) + 1}): ").strip()
            
            if choice == str(len(available_diseases) + 1):
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_diseases):
                selected = available_diseases[choice_idx]
                print(f"Selected: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(available_diseases) + 1}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

# Orchestrator function
async def run_epidemic_simulation(selected_disease: str = None):
    """Run the complete epidemic simulation with agno agents"""
    
    try:
        # Load actual data
        df = load_existing_data()
        
        if df.empty:
            return {"error": "No data available from database"}
        
        # Get available diseases
        available_diseases = df['disease'].unique().tolist() if 'disease' in df.columns else []
        
        print(f"Loaded data with {len(df)} records")
        print(f"Available diseases: {available_diseases}")
        
        # Stage 1: Data Drift Analysis
        print("\n" + "="*60)
        print("STAGE 1: DATA DRIFT ANALYSIS")
        print("="*60)
        
        drift_task = f"""
        Analyze the epidemic surveillance data for data drift patterns. 
        
        Data summary:
        - Total records: {len(df)}
        - Available diseases: {available_diseases}
        - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}

        Focus on identifying shifts in:
        - Disease patterns over time
        - Patient demographics and case distributions
        - Geographic distribution patterns
        - Data quality and completeness
        
        Provide a comprehensive analysis of data quality and any concerning drift patterns.
        Use statistical methods to detect changes between recent and historical data.
        """
        
        # Use synchronous call instead of async - Agno agents use .print_response() not .run()
        print("Running data drift analysis...")
        drift_response: RunResponse = data_drift_agent.run(drift_task)
        drift_result = drift_response.content
        data_drift_agent.print_response(drift_task, stream=True)
        
        # Stage 2: Disease Spread Simulation (if disease selected)
        if selected_disease and selected_disease in available_diseases:
            print("\n" + "="*60)
            print(f"STAGE 2: DISEASE SPREAD SIMULATION FOR {selected_disease}")
            print("="*60)
            
            # Filter data for selected disease
            disease_data = df[df['disease'] == selected_disease]

            # Extract numeric part safely
            disease_data.loc[:, 'case_id_numeric'] = pd.to_numeric(
                disease_data['case_id'].str.extract(r'(\d+)')[0],
                errors='coerce'
            )

            # Format average and max safely
            avg_case_id = (
                f"{disease_data['case_id_numeric'].mean():.1f}"
                if disease_data['case_id_numeric'].notna().any()
                else "N/A"
            )

            max_case_id = (
                int(disease_data['case_id_numeric'].max())
                if disease_data['case_id_numeric'].notna().any()
                else "N/A"
            )
            
            simulation_task = f"""
            Simulate the spread of {selected_disease} using the epidemic surveillance data.
            
            Disease-specific data summary:
            - {selected_disease} records: {len(disease_data)}
            - Average case_id number: {avg_case_id}
            - Peak case_id number: {max_case_id}
            - Regions affected: {len(disease_data['region'].unique()) if 'region' in disease_data.columns else 'N/A'}
            - interventions: {disease_data['intervention_type'].unique()}
            
            Run a SEIR model simulation (Do not create code) with:
            - Simulation period: 90 days
            - Multiple scenarios (optimistic, realistic, pessimistic)
            - Parameter estimation based on observed data patterns
            
            Provide insights about expected peak infections, total cases, and timeline.
            Include confidence intervals and sensitivity analysis.
            """
            
            print(f"Running disease simulation for {selected_disease}...")
            simulation_response: RunResponse = disease_simulation_agent.run(simulation_task)
            simulation_result = simulation_response.content
            disease_simulation_agent.print_response(simulation_task, stream=True)
            
            # Stage 3: Intervention Evaluation
            print("\n" + "="*60)
            print(f"STAGE 3: INTERVENTION EVALUATION FOR {selected_disease}")
            print("="*60)
            
            intervention_task = f"""
            Evaluate intervention strategies for {selected_disease} based on available data and literature.
            
            Context:
            - Disease: {selected_disease}
            - Overall simulation from Disease Spread Agent: {simulation_result}

            Analyze and rank interventions from the given simulation results.
            
            Provide evidence-based recommendations with:
            - Effectiveness rankings
            - Cost-benefit analysis
            - Implementation feasibility
            - Expected impact on transmission
            """
            
            print(f"Running intervention evaluation for {selected_disease}...")
            intervention_response: RunResponse = intervention_agent.run(intervention_task)
            intervention_result = intervention_response.content
            intervention_agent.print_response(intervention_task, stream=True)

        else:
            simulation_result = "No disease selected - skipped simulation"
            intervention_result = "No disease selected - skipped intervention analysis"

        print("Drift Report:", drift_result)
        print("Simulation Report:", simulation_result)
        print("Intervention Report:", intervention_result)

        results = {
            'available_diseases': available_diseases,
            'selected_disease': selected_disease,
            'data_summary': {
                'total_records': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'diseases': available_diseases
            },
            'drift_analysis_completed': True,
            'simulation_completed': selected_disease is not None,
            'intervention_analysis_completed': selected_disease is not None,
            'drift_report': drift_result,  # Capture report text
            'simulation_report': simulation_result if selected_disease else None,  # Capture report text
            'intervention_report': intervention_result if selected_disease else None,  # Capture report text
            'timestamp': datetime.now().isoformat()
        }
        
        if results.get('drift_report'):
            report_metadata = {
                'title': 'Epidemic Analysis Report',
                'subtitle': 'Data Drift Analysis Report', # The H1 on the first page
                'author': 'DataDriftAnalyst',
                'division': 'Epidemiological Data Quality Division',
                'date': datetime.now().isoformat(),
                'generation_date': datetime.now().isoformat()
            }
            create_final_report(results['drift_report'], 'Drift_Analysis_Report.pdf', report_metadata)
        if results.get('simulation_report'):
            report_metadata = {
                'title': 'Epidemic Analysis Report',
                'subtitle': f'Disease Simulation Report {selected_disease}', # The H1 on the first page
                'author': 'DiseaseSimulation',
                'division': 'Epidemiological Data Quality Division',
                'date': datetime.now().isoformat(),
                'generation_date': datetime.now().isoformat()
            }
            create_final_report(results['simulation_report'], f'Disease_Simulation_Report_{selected_disease}.pdf', report_metadata)
        if results.get('intervention_report'):
            report_metadata = {
                'title': 'Epidemic Analysis Report',
                'subtitle': f'Intervention Evaluation Report {selected_disease}', # The H1 on the first page
                'author': 'InterventionEvaluationAnalyst',
                'division': 'Epidemiological Data Quality Division',
                'date': datetime.now().isoformat(),
                'generation_date': datetime.now().isoformat()
            }
            create_final_report(results['intervention_report'], f'Intervention_Evaluation_Report_{selected_disease}.pdf', report_metadata)
        
        return results
        
    except Exception as e:
        print(f"Error in epidemic simulation: {str(e)}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

def main():
    """Main function to run the epidemic analysis with user interaction"""
    print("="*80)
    print("EPIDEMIC SURVEILLANCE ANALYSIS SYSTEM")
    print("="*80)
    
    try:
        # Step 1: Load data and get available diseases
        df = load_existing_data()
        available_diseases = df['disease'].unique().tolist() if 'disease' in df.columns else []
        
        if not available_diseases:
            print("No diseases found in the data. Exiting...")
            return
        
        # Step 2: Get user's disease selection
        selected_disease = get_user_disease_selection(available_diseases)
        
        # Step 3: Run the analysis
        print(f"\nStarting epidemic analysis...")
        if selected_disease:
            print(f"Selected disease: {selected_disease}")
        else:
            print("Running general analysis without disease-specific simulation")
        
        # Note: Agno agents are synchronous, so we don't use asyncio.run()
        # Instead, we call the function directly
        results = asyncio.run(run_epidemic_simulation(selected_disease=selected_disease))
        
        # Step 4: Display results summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Analysis completed at: {results.get('timestamp', 'Unknown')}")
        print(f"Data processed: {results.get('data_summary', {}).get('total_records', 'Unknown')} records")
        
        if selected_disease:
            print(f"Disease analyzed: {selected_disease}")
            print("✓ Data drift analysis completed")
            print("✓ Disease spread simulation completed")
            print("✓ Intervention evaluation completed")
        else:
            print("✓ Data drift analysis completed")
            print("- Disease simulation skipped (no disease selected)")
            print("- Intervention analysis skipped (no disease selected)")
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError running analysis: {str(e)}")

if __name__ == "__main__":
    main()