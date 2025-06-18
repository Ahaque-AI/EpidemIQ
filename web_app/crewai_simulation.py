import crewai
from neo4j import GraphDatabase
import os
import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from tavily import TavilyClient
from crewai.tools import tool
import requests
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and completeness."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, disease_name: str) -> Tuple[bool, List[str]]:
        """Validate dataframe for simulation requirements."""
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check for required columns
        required_columns = ['case_id', 'timestamp', 'disease', 'region', 'severity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check disease data
        disease_data = df[df['disease'].str.lower() == disease_name.lower()]
        if disease_data.empty:
            issues.append(f"No data found for disease: {disease_name}")
        elif len(disease_data) < 10:
            issues.append(f"Insufficient data for {disease_name}: only {len(disease_data)} cases")
        
        # Check data quality
        if df['timestamp'].isna().sum() > df.shape[0] * 0.1:
            issues.append("More than 10% of timestamps are missing")
        
        return len(issues) == 0, issues


class EnhancedDiseaseSimulation:
    """Enhanced disease simulation with robust error handling and improved prompts."""
    
    def __init__(self, df: pd.DataFrame, disease_name: str, simulation_days: int = 30):
        self.df = df
        self.disease_name = disease_name.lower()
        self.simulation_days = max(1, min(simulation_days, 365))  # Limit simulation days
        self.filtered_df = df[df['disease'].str.lower() == self.disease_name]
        
        # Initialize LLM with error handling
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.llm = ChatGroq(
            api_key=api_key,
            temperature=0.3,
            model_name="groq/llama-3.1-8b-instant",
            max_tokens=4000
        )
        
        # Validate data
        is_valid, issues = DataValidator.validate_dataframe(df, disease_name)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        logger.info(f"Initialized simulation for {disease_name} with {len(self.filtered_df)} cases over {simulation_days} days")

    def generate_data_summary(self) -> Dict:
        """Generate comprehensive data summary for agents."""
        if self.filtered_df.empty:
            return {"error": "No data available for the specified disease"}
        
        summary = {
            "total_cases": len(self.filtered_df),
            "date_range": {
                "start": self.filtered_df['timestamp'].min().strftime('%Y-%m-%d') if not self.filtered_df['timestamp'].isna().all() else "Unknown",
                "end": self.filtered_df['timestamp'].max().strftime('%Y-%m-%d') if not self.filtered_df['timestamp'].isna().all() else "Unknown"
            },
            "demographics": {
                "age_distribution": self.filtered_df['age'].describe().to_dict() if 'age' in self.filtered_df.columns else {},
                "gender_distribution": self.filtered_df['gender'].value_counts().to_dict() if 'gender' in self.filtered_df.columns else {},
                "age_group_distribution": self.filtered_df['age_group'].value_counts().to_dict() if 'age_group' in self.filtered_df.columns else {}
            },
            "geographic": {
                "region_distribution": self.filtered_df['region'].value_counts().to_dict() if 'region' in self.filtered_df.columns else {},
                "unique_regions": self.filtered_df['region'].nunique() if 'region' in self.filtered_df.columns else 0
            },
            "clinical": {
                "severity_distribution": self.filtered_df['severity'].value_counts().to_dict() if 'severity' in self.filtered_df.columns else {},
                "hospitalization_rate": self.filtered_df['hospitalization_required'].mean() if 'hospitalization_required' in self.filtered_df.columns else 0,
                "contact_tracing_rate": self.filtered_df['contact_tracing_needed'].mean() if 'contact_tracing_needed' in self.filtered_df.columns else 0
            },
            "interventions": {
                "types": self.filtered_df['intervention_type'].value_counts().to_dict() if 'intervention_type' in self.filtered_df.columns else {},
                "effectiveness_stats": self.filtered_df['intervention_effectiveness'].describe().to_dict() if 'intervention_effectiveness' in self.filtered_df.columns else {},
                "compliance_stats": self.filtered_df['compliance_rate'].describe().to_dict() if 'compliance_rate' in self.filtered_df.columns else {}
            }
        }
        
        return summary

    def create_agents(self):
        """Create enhanced agents with more robust prompts."""
        
        data_summary = self.generate_data_summary()
        
        # Agent 1: Enhanced Pattern Identifier
        pattern_identifier = Agent(
            role='Senior Epidemiological Pattern Analyst',
            goal=f'Conduct comprehensive analysis of {self.disease_name} transmission patterns and risk factors using advanced epidemiological methods',
            backstory=f"""You are a world-renowned epidemiologist with 20+ years of experience in disease pattern analysis and outbreak investigation.
            
            Your expertise includes:
            - Advanced statistical modeling of disease transmission
            - Risk factor identification and stratification
            - Geographic disease mapping and cluster analysis
            - Intervention effectiveness evaluation
            - Predictive modeling for outbreak scenarios
            
            You have access to comprehensive data for {self.disease_name} containing {data_summary.get('total_cases', 0)} cases.
            Your analysis will form the foundation for accurate disease spread simulation.""",
            allow_delegation=False,
            llm=ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,
                model_name="groq/llama-3.1-8b-instant",
                max_tokens=4000
            ),
            verbose=True,
            max_iter=1,
            max_retry_limit=1
        )

        # Agent 2: Enhanced Simulation Engine
        simulation_engine = Agent(
            role='Advanced Disease Modeling Specialist',
            goal=f'Develop and execute sophisticated {self.disease_name} transmission models with realistic population dynamics',
            backstory=f"""You are a computational epidemiologist specializing in disease transmission modeling with expertise in:
            
            - Stochastic epidemic modeling (SIR, SEIR, compartmental models)
            - Population dynamics and mobility patterns
            - Intervention impact modeling
            - Monte Carlo simulation techniques
            - Real-time adaptive modeling based on emerging data
            
            You will create a {self.simulation_days}-day simulation that accounts for:
            - Variable transmission rates based on demographic and geographic factors
            - Realistic intervention implementation with compliance variation
            - Healthcare system capacity constraints
            - Population behavioral responses to interventions
            - Stochastic elements to reflect real-world uncertainty
            - You have access to internet through two tools, you can learn how to simulate data""",
            allow_delegation=False,
            llm=ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,
                model_name="groq/meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=4000
            ),
            verbose=True,
            max_iter=1,
            max_retry_limit=1
        )

        # Agent 3: Enhanced Report Compiler
        report_compiler = Agent(
            role='Medical Intelligence Report Specialist',
            goal='Synthesize simulation results into actionable intelligence reports for public health decision-makers',
            backstory=f"""You are a medical intelligence analyst with extensive experience in:
            
            - Public health emergency response reporting
            - Statistical analysis and data visualization
            - Risk communication and policy recommendations
            - Intervention cost-effectiveness analysis
            - Uncertainty quantification and confidence intervals
            
            Your reports are used by:
            - Public health officials for policy decisions
            - Healthcare administrators for resource allocation
            - Emergency response teams for outbreak preparedness
            - Researchers for scientific publication
            
            You excel at translating complex epidemiological data into clear, actionable insights.""",
            allow_delegation=False,
            llm=ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,
                model_name="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
                max_tokens=4000
            ),
            verbose=True,
            max_iter=1,
            max_retry_limit=1
        )

        return pattern_identifier, simulation_engine, report_compiler

    def create_tasks(self, pattern_identifier, simulation_engine, report_compiler):
        """Create enhanced tasks with comprehensive instructions."""
        
        data_summary = self.generate_data_summary()
        
        # Task 1: Enhanced Pattern Identification
        pattern_identification = Task(
            description=f"""Conduct a comprehensive epidemiological analysis of {self.disease_name} using the provided dataset.

            ## DATASET OVERVIEW:
            {json.dumps(data_summary, indent=2)}

            ## PAST HISTORY:
            {self.filtered_df}
            {self.filtered_df['intervention_type'].unique()}

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>


            ## ANALYSIS REQUIREMENTS:

            ### 1. TRANSMISSION DYNAMICS ANALYSIS
            - Calculate basic reproduction number (R0) estimates by region and time period
            - Identify superspreading events and transmission hotspots
            - Analyze generation time and serial interval patterns
            - Determine seasonal/temporal transmission variations

            ### 2. DEMOGRAPHIC RISK STRATIFICATION
            - Age-specific attack rates and severity patterns
            - Gender-based susceptibility and outcome differences
            - Comorbidity and risk factor correlations
            - High-risk population identification

            ### 3. GEOGRAPHIC SPREAD PATTERNS
            - Regional transmission velocity and direction
            - Urban vs rural spread differences
            - Transportation corridor impact analysis
            - Border and mobility-related transmission patterns

            ### 4. INTERVENTION EFFECTIVENESS QUANTIFICATION
            - Intervention-specific effectiveness scores with confidence intervals
            - Time-to-effect analysis for different intervention types
            - Compliance rate impact on effectiveness
            - Cost-effectiveness ratios for each intervention

            ### 5. OUTBREAK PREDICTION INDICATORS
            - Early warning signals and threshold identification
            - Leading indicators for outbreak escalation
            - Geographic expansion prediction models
            - Healthcare system strain predictors

            ## OUTPUT REQUIREMENTS:
            Provide quantitative metrics with statistical confidence levels for all findings.
            Include specific numerical values, rates, and ratios that can be used in simulation modeling.
            Identify the top 5 most critical factors for disease spread prediction.""",
            
            agent=pattern_identifier,
            expected_output=f"""COMPREHENSIVE {self.disease_name.upper()} EPIDEMIOLOGICAL ANALYSIS

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>

            ## EXECUTIVE SUMMARY
            [Key findings with specific numerical values]

            ## TRANSMISSION DYNAMICS
            - R0 estimates: [value ± confidence interval] by region
            - Generation time: [mean ± SD] days
            - Superspreading threshold: [specific criteria]

            ## RISK FACTORS (Ranked by impact)
            1. [Factor]: Risk ratio [value], Confidence interval [range]
            2. [Factor]: Risk ratio [value], Confidence interval [range]
            [Continue for top 10 factors]

            ## GEOGRAPHIC PATTERNS
            - Highest risk regions: [list with transmission rates]
            - Spread velocity: [km/day or similar metric]
            - Critical transmission corridors: [specific routes/areas]

            ## INTERVENTION EFFECTIVENESS MATRIX
            [Table format with intervention type, effectiveness %, compliance rate, cost-effectiveness ratio]

            ## OUTBREAK PREDICTION MODEL
            - Early warning indicators: [specific metrics and thresholds]
            - Escalation probability formula: [mathematical expression]
            - Healthcare capacity thresholds: [specific bed/ICU ratios]
            
            ## PAST HISTORY:
            {self.filtered_df}
            {self.filtered_df['intervention_type'].unique()}
            """,
            output_file="output/pattern_identification.md"
        )

        # Task 2: Enhanced Simulation Execution
        simulation_execution = Task(
            description=f"""Execute a sophisticated {self.simulation_days}-day disease transmission simulation for {self.disease_name}.

            ## SIMULATION PARAMETERS:
            - Initial conditions: {len(self.filtered_df)} confirmed cases
            - Simulation period: {self.simulation_days} days
            - Population dynamics: Include realistic mobility and mixing patterns
            - Stochastic elements: Include random variation in transmission

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>

            ## MODELING REQUIREMENTS:

            ### 1. TRANSMISSION MODEL
            - Implement compartmental model (SEIR or appropriate variant)
            - Use region-specific transmission rates from pattern analysis
            - Include demographic-specific susceptibility factors
            - Account for behavioral changes during outbreak progression

            ### 2. INTERVENTION IMPLEMENTATION
            - Trigger interventions when daily new cases exceed 5% of active cases OR when regional R_effective > 1.5
            - Model realistic implementation delays (1-7 days depending on intervention type)
            - Include compliance variation based on historical data
            - Account for intervention fatigue over time

            ### 3. HEALTHCARE SYSTEM MODELING
            - Track hospital bed utilization (general and ICU)
            - Model healthcare worker infections and availability
            - Include testing capacity constraints
            - Account for treatment availability and effectiveness

            ### 4. POPULATION DYNAMICS
            - Include realistic contact patterns by age group and setting
            - Model population mobility and mixing changes
            - Account for economic and social factors affecting behavior
            - Include seasonal variations if applicable

            ### 5. UNCERTAINTY QUANTIFICATION
            - Run multiple simulation iterations (minimum 100)
            - Provide confidence intervals for all projections
            - Include sensitivity analysis for key parameters
            - Quantify model uncertainty and limitations

            ## DAILY TRACKING REQUIREMENTS:
            For each day, track and report:
            - New cases (suspected, probable, confirmed)
            - Active cases by severity level
            - Hospitalizations (general ward, ICU, ventilator)
            - Deaths and recoveries
            - Tests performed and positivity rate
            - Interventions active and compliance levels
            - Healthcare system utilization percentages
            - Economic impact indicators""",
            
            agent=simulation_engine,
            expected_output=f"""COMPREHENSIVE {self.simulation_days}-DAY {self.disease_name.upper()} SIMULATION RESULTS

            ## SIMULATION SUMMARY
            - Total projected cases: [number with 95% CI]
            - Peak daily incidence: [number] on day [X]
            - Attack rate: [percentage] of population
            - Case fatality rate: [percentage with CI]

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>

            ## DAILY SIMULATION DATA
            - add explanation of what the columns tell as well.
            [Table with columns: Day, New_Cases, Active_Cases, Hospitalizations, Deaths, Interventions_Active, R_effective, Healthcare_Utilization_%]

            ## INTERVENTION TIMELINE
            Day [X]: [Intervention] implemented, compliance [%], estimated effect size [value]
            [Continue for all interventions]

            ## HEALTHCARE SYSTEM IMPACT
            - Peak bed utilization: [%] on day [X]
            - Peak ICU utilization: [%] on day [X]
            - Healthcare worker infection rate: [%]
            - Critical resource shortages: [list with timing]

            ## UNCERTAINTY ANALYSIS
            - Model confidence score: [0-1] with justification
            - Key uncertainty sources: [ranked list]
            - Sensitivity analysis results: [parameter impacts]

            ## SCENARIO COMPARISONS
            - No intervention scenario: [key metrics]
            - Optimal intervention scenario: [key metrics]
            - Resource-constrained scenario: [key metrics]""",
            context=[pattern_identification],
            output_file="output/simulation.md"
        )

        # Task 3: Enhanced Report Compilation
        report_compilation = Task(
            description=f"""Compile a comprehensive public health intelligence report on the {self.disease_name} simulation results.

            ## REPORT STRUCTURE REQUIREMENTS:

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>

            ### 1. EXECUTIVE SUMMARY (1-2 pages)
            - Key findings in bullet points
            - Critical decision points and recommendations
            - Resource requirements and timeline
            - Risk assessment with confidence levels

            ### 2. METHODOLOGY SECTION
            - Data sources and quality assessment
            - Model assumptions and limitations
            - Validation approach and confidence metrics
            - Uncertainty quantification methods

            ### 3. DETAILED RESULTS
            - Statistical tables with confidence intervals
            - Time series plots description
            - Geographic heat maps interpretation
            - Demographic breakdown analysis

            ### 4. INTERVENTION ANALYSIS
            - Cost-effectiveness evaluation for each intervention
            - Optimal intervention timing and sequencing
            - Resource allocation recommendations
            - Implementation feasibility assessment

            ### 5. RISK ASSESSMENT
            - Probability of different outbreak scenarios
            - Healthcare system breaking points
            - Economic impact projections
            - Social disruption risk factors

            ### 6. RECOMMENDATIONS
            - Immediate actions (next 7 days)
            - Short-term strategy (next 30 days)
            - Long-term preparedness improvements
            - Research and surveillance priorities

            ## QUALITY STANDARDS:
            - All claims must be supported by specific data
            - Include uncertainty ranges for all projections
            - Provide actionable recommendations with clear timelines
            - Format for both technical and executive audiences""",
            
            agent=report_compiler,
            expected_output=f"""# {self.disease_name.upper()}

            # Add in all the information from the context of the two agents and then write REPORT. 
            # The REPORT should be very detailed and minimum 3 pages.
             
            OUTBREAK SIMULATION INTELLIGENCE REPORT

            ## EXECUTIVE SUMMARY

            ### Key Findings
            - **Outbreak Scale**: [projected total cases with CI]
            - **Timeline**: Peak expected on day [X], duration [Y] days
            - **Healthcare Impact**: [peak utilization] requiring [specific resources]
            - **Intervention Effectiveness**: [most effective intervention] reduces cases by [%]

            <ImportantRule> If you create any table you should create it properly with proper rows and columns seperated using | or --. like this:
                | Intervention Type | Effectiveness (%) | Compliance Rate (%) | Cost-Effectiveness Ratio |
                |-------------------|-------------------|---------------------|--------------------------|
                | Mask Mandate      | 85.2              | 92.1                | 1.23                     |
                | Travel Restriction| 78.5              | 85.6                | 1.51                     |
                | Contact Tracing   | 75.1              | 83.2                | 1.63                     |
                | Social Distancing | 72.5              | 80.9                | 1.73                     |
            <ImportantRule>

            ### Critical Recommendations
            1. **IMMEDIATE** (Next 7 days): [specific actions]
            2. **SHORT-TERM** (Next 30 days): [strategic interventions]
            3. **LONG-TERM**: [preparedness improvements]

            ## METHODOLOGY
            [Detailed methodology with validation approach]

            ## SIMULATION RESULTS

            ### Overall Outbreak Trajectory
            [Comprehensive table with daily projections including confidence intervals]

            ### Intervention Effectiveness Analysis
            | Intervention | Effectiveness % | Cost per Case Prevented | Implementation Time | Compliance Rate |
            |--------------|-----------------|-------------------------|---------------------|-----------------| 
            [Complete intervention matrix]

            ### Geographic Risk Assessment
            [Regional breakdown with risk levels and resource needs]

            ### Healthcare System Impact
            [Detailed capacity analysis with breaking points]

            ## RISK ASSESSMENT

            ### Scenario Probabilities
            - Best case (95% CI): [metrics]
            - Most likely (50% CI): [metrics] 
            - Worst case (5% CI): [metrics]

            ### Confidence Assessment
            - Model reliability score: [0-10] based on [criteria]
            - Key uncertainties: [ranked list with impact assessment]

            ## ACTIONABLE RECOMMENDATIONS

            ### Immediate Actions (0-7 days)
            [Specific, time-bound recommendations]

            ### Strategic Interventions (7-30 days)
            [Medium-term planning recommendations]

            ### Preparedness Improvements (30+ days)
            [Long-term system strengthening]

            ## APPENDICES
            - Technical methodology details
            - Data quality assessment
            - Model validation results
            - Sensitivity analysis tables""",
            context=[pattern_identification, simulation_execution],
            output_file="output/Report.md"
        )

        return [pattern_identification, simulation_execution, report_compilation]

    def run_simulation(self) -> str:
        """Execute the complete simulation workflow with enhanced error handling."""
        try:
            logger.info(f"Starting {self.disease_name} simulation for {self.simulation_days} days")
            
            if self.filtered_df.empty:
                return f"ERROR: No data found for disease '{self.disease_name}'. Available diseases: {self.df['disease'].unique().tolist()}"
            
            # Create agents and tasks
            logger.info("Creating agents and tasks...")
            pattern_identifier, simulation_engine, report_compiler = self.create_agents()
            tasks = self.create_tasks(pattern_identifier, simulation_engine, report_compiler)

            # Create and run the crew
            logger.info("Initializing CrewAI workflow...")
            crew = Crew(
                agents=[pattern_identifier, simulation_engine, report_compiler],
                tasks=tasks,
                verbose=True,
                process=Process.sequential,
                memory=True,
                embedder={
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                }
            )

            logger.info("Executing simulation workflow...")
            result = crew.kickoff()
            
            logger.info("Simulation completed successfully")
            return str(result)

        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

def run(disease_input, days_input, df):
    """Main execution function with comprehensive error handling."""
    try:
        
        # Load data
        logger.info("Loaded Data......")
        
        # Display available diseases
        available_diseases = df['disease'].value_counts()
        logger.info(f"Available diseases: {available_diseases.to_dict()}")
        
        # Configure simulation parameters
        disease_name = disease_input  # Change this to your target disease
        simulation_days = days_input
        
        # Validate disease exists in data
        if disease_name.lower() not in df['disease'].str.lower().values:
            logger.error(f"Disease '{disease_name}' not found in data. Available: {df['disease'].unique().tolist()}")
            return
        
        # Run simulation
        logger.info(f"Starting simulation for {disease_name}...")
        simulation = EnhancedDiseaseSimulation(
            df=df,
            disease_name=disease_name,
            simulation_days=simulation_days
        )
        
        report = simulation.run_simulation()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{disease_name}_simulation_report_{timestamp}.md"
        
        with open("output/Report.md", 'r', encoding='utf-8') as f:
            report_content = f.read()

        with open("output/simulation.md", 'r', encoding='utf-8') as f:
            simulation_content = f.read()

        with open("output/pattern_identification.md", 'r', encoding='utf-8') as f:
            pattern_content = f.read()

        final_report = "\n\n## Pattern Identification\n" + pattern_content +  "\n\n## Simulation Results\n" + simulation_content + "\n\n## Full Report\n" + report_content

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        logger.info(f"Report saved to {filename}")
        print(f"\nSimulation completed successfully. Report saved to: {filename}")
        print(f"\nReport preview:\n{final_report[:1000]}...")

        return filename
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise