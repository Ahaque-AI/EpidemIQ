import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
from flask import jsonify, request
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

llm = ChatNVIDIA(
    model_name="nvidia/llama-3.1-405b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

report_prompt_template = PromptTemplate(
    input_variables=["data_summary", "analysis_focus", "report_type"],
    template="""
    You are a senior epidemiologist and public health expert analyzing disease outbreak data. 
    
    **DATASET OVERVIEW:**
    {data_summary}
    
    **ANALYSIS FOCUS:** {analysis_focus}
    **REPORT TYPE:** {report_type}
    
    **INSTRUCTIONS:**
    1. Analyze the provided healthcare data comprehensively
    2. Identify critical patterns, trends, and anomalies
    3. Provide actionable insights for public health decision-making
    4. Include risk assessments and recommendations
    5. Use medical terminology appropriately while keeping it accessible
    6. Structure your response professionally
    
    **REPORT STRUCTURE:**
    - Executive Summary (key findings in 2-3 sentences)
    - Critical Risk Factors Identified
    - Demographic Analysis & Vulnerable Populations
    - Geographic Distribution & Hotspots
    - Temporal Trends & Seasonality
    - Disease Progression Patterns
    - Intervention Effectiveness Assessment
    - Strategic Recommendations
    - Immediate Action Items
    
    **FOCUS AREAS:**
    - Highlight the most severe cases and their characteristics
    - Identify populations at highest risk
    - Recommend targeted intervention strategies
    - Suggest resource allocation priorities
    - Provide early warning indicators for future outbreaks

    Note: do not include any emojis or symbols in the report. even em dashes or quotes.
    
    Generate a comprehensive, professional public health report based on this data.
    Aim for ~1200-2000 words total. If any critical fields are missing from the overview (e.g., no `timestamp`), please note these gaps and their potential impact on the analysis.
    """
)


# Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jDataDriftAnalyzer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.baseline_data = None
        self.current_data = None
        
    def close(self):
        self.driver.close()
    
    def extract_all_data(self):
        """Extract all case data from Neo4j into a DataFrame"""
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
        
        with self.driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
        
        df = pd.DataFrame(records)
        if not df.empty:
            # Convert Neo4j DateTime to Python datetime and remove timezone info
            def convert_timestamp(x):
                if hasattr(x, 'to_native'):
                    dt = x.to_native()
                    # Remove timezone info if present
                    return dt.replace(tzinfo=None) if dt.tzinfo else dt
                else:
                    # Handle regular datetime objects
                    return x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x
            
            df['timestamp'] = df['timestamp'].apply(convert_timestamp)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symptoms_count'] = df['symptoms'].apply(len)
        
        return df
    
Neo4jDataDriftAnalyzer = Neo4jDataDriftAnalyzer()
df = Neo4jDataDriftAnalyzer.extract_all_data()

# save inside json file 'all_data.json'
df.to_json('all_data.json', orient='records', date_format='iso', indent=2)

df = pd.read_json('all_data.json', orient='records', convert_dates=['timestamp'])

def prepare_dataframe_summary(df: pd.DataFrame) -> dict:
    """Generate a comprehensive summary of a health-records DataFrame."""
    summary = {}

    # 1. Schema & Missingness
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        try:
            unique = int(df[col].nunique(dropna=True))
        except TypeError:
            unique = int(
                df[col]
                .dropna()
                .apply(lambda x: tuple(x) if isinstance(x, list) else x)
                .nunique()
            )
        missing = int(df[col].isna().sum())
        pct_missing = float(df[col].isna().mean() * 100)
        schema[col] = {
            "dtype": dtype,
            "unique_values": unique,
            "missing_count": missing,
            "missing_pct": round(pct_missing, 2)
        }
    summary['schema'] = schema

    # 2. Basic Counts & Date Range
    summary['basic'] = {
        "total_records": len(df),
        "date_min": None,
        "date_max": None
    }
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        summary['basic']['date_min'] = ts.min().isoformat() if not ts.isna().all() else None
        summary['basic']['date_max'] = ts.max().isoformat() if not ts.isna().all() else None

    # 3. Temporal Trends
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
        monthly = ts.dt.to_period('M').value_counts().sort_index()
        summary['temporal_trends'] = {
            "monthly_counts": {str(idx): int(cnt) for idx, cnt in monthly.items()}
        }

    # 4. Numeric Overview & Correlations
    num = df.select_dtypes(include=[np.number])
    stats = {}
    for col in num.columns:
        stats[col] = {
            "count": int(num[col].count()),
            "mean": float(num[col].mean()),
            "median": float(num[col].median()),
            "std": float(num[col].std()),
            "min": float(num[col].min()),
            "max": float(num[col].max()),
        }
    summary['numeric_stats'] = stats

    if not num.empty:
        corr = num.corr().abs()
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
        )
        top_corrs = corr_pairs.head(5)
        summary['top_numeric_correlations'] = [
            {"pair": f"{i[0]}↔{i[1]}", "corr": round(v, 2)}
            for i, v in top_corrs.items()
        ]

    # 5. Categorical Distributions
    cat = df.select_dtypes(include=['object', 'category'])
    cat_dist = {}
    for col in cat.columns:
        counts = cat[col].value_counts(dropna=False).head(10)
        total = len(df[col])
        dist = {}
        for k, v in counts.items():
            if isinstance(k, list):
                key = str(tuple(k))
            elif pd.isna(k):
                key = "NaN"
            else:
                key = str(k)
            dist[key] = {
                "count": int(v),
                "pct": round(v/total*100, 2)
            }
        cat_dist[col] = dist
    summary['categorical_distributions'] = cat_dist

    # 6. Geospatial Overview
    if {'latitude', 'longitude'}.issubset(df.columns):
        valid = df[['latitude', 'longitude']].dropna().astype(float)
        summary['geo'] = {
            "lat_min": valid['latitude'].min(),
            "lat_max": valid['latitude'].max(),
            "lon_min": valid['longitude'].min(),
            "lon_max": valid['longitude'].max(),
            "center": {
                "latitude": float(valid['latitude'].mean()),
                "longitude": float(valid['longitude'].mean())
            }
        }
        if 'region' in df.columns:
            top_regions = df['region'].value_counts().head(5)
            summary['geo']['top_regions'] = {
                str(idx): int(cnt) for idx, cnt in top_regions.items()
            }

    # 7. Intervention Analysis
    if 'intervention_type' in df.columns:
        grp = df.groupby('intervention_type')
        intervention = {}
        for name, g in grp:
            intervention[name] = {
                "count": int(len(g)),
                "avg_effectiveness": float(g['intervention_effectiveness'].mean()) if 'intervention_effectiveness' in g else None,
                "avg_cost": float(g['intervention_cost'].mean()) if 'intervention_cost' in g else None,
                "avg_compliance": float(g['compliance_rate'].mean()) if 'compliance_rate' in g else None
            }
        summary['intervention_summary'] = intervention

    # 8. Outbreak & Severity Insights
    if 'is_outbreak_related' in df.columns:
        outbreak = df['is_outbreak_related'].sum()
        summary['outbreak'] = {
            "outbreak_related_count": int(outbreak),
            "outbreak_pct": round(outbreak/len(df)*100, 2)
        }
    if 'severity' in df.columns:
        # coerce severity to numeric to handle string/noisy entries
        sev = pd.to_numeric(df['severity'], errors='coerce')
        quantile_80 = sev.quantile(0.8)
        high = int((sev > quantile_80).sum())
        summary['severity_insights'] = {
            "high_severity_count": high,
            "high_severity_pct": round(high/len(df)*100, 2),
            "avg_severity": float(sev.mean()) if not sev.isna().all() else None
        }

    # 9. Symptom Summary
    if 'symptoms' in df.columns:
        all_sym = df['symptoms'].dropna().apply(lambda x: x.split(',') if isinstance(x, str) else list(x))
        flat = pd.Series([s.strip() for sub in all_sym for s in sub])
        top_sym = flat.value_counts().head(10)
        summary['symptom_summary'] = {
            "unique_symptoms": int(flat.nunique()),
            "top_symptoms": {k: int(v) for k, v in top_sym.items()},
            "avg_symptom_count": float(all_sym.apply(len).mean())
        }

    return summary

def generate_ai_report():
    """Generate AI-powered health report using NVIDIA LLM"""
    # Get request parameters
    analysis_focus = 'comprehensive_analysis'
    report_type = 'epidemiological_report'
    
    # Access the global dataframe
    global df
    
    # Prepare data summary for LLM
    data_summary = prepare_dataframe_summary(df)
    
    # Convert summary to JSON string for the prompt
    data_summary_str = json.dumps(data_summary, indent=2, default=str)
    
    # Create the prompt
    prompt = report_prompt_template.format(
        data_summary=data_summary_str,
        analysis_focus=analysis_focus,
        report_type=report_type
    )
    
    # Generate report using NVIDIA LLM
    print("Generating AI report...")  # Debug log
    response = llm.invoke(prompt)
    
    report_content = response.content if hasattr(response, 'content') else str(response)
    
    # Prepare response
    report_data = {
        "report": report_content,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "data_records_analyzed": len(df),
            "analysis_focus": analysis_focus,
            "report_type": report_type,
            "model_used": "nvidia/llama-3.1-405b-instruct"
        },
        "status": "success"
    }
    
    return report_data
    
AIReport = generate_ai_report()
# convert the AIReport to pdf but good format and save it as 'health_report.pdf'

def save_report_as_pdf(report_content, filename='health_report.pdf'):
    """Save the AI-generated report content as a professionally formatted PDF file."""
    from fpdf import FPDF
    from datetime import datetime
    import re
    
    class HealthReportPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.report_title = "Epidemiological Report: Comprehensive Analysis of Healthcare Data"
            
        def header(self):
            if self.page_no() > 1:  # Skip header on title page
                # Set header font
                self.set_font('Arial', 'B', 10)
                self.set_text_color(70, 70, 70)
                
                # Add title in header
                self.cell(0, 10, self.report_title, 0, 1, 'C')
                
                # Add line under header
                self.set_draw_color(200, 200, 200)
                self.line(10, 25, 200, 25)
                self.ln(5)
        
        def footer(self):
            if self.page_no() > 1:  # Skip footer on title page
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                
                # Add line above footer
                self.set_draw_color(200, 200, 200)
                self.line(10, self.get_y() - 5, 200, self.get_y() - 5)
                
                # Set footer font
                self.set_font('Arial', 'I', 8)
                self.set_text_color(128, 128, 128)
                
                # Page number (starts from 1 on first content page)
                self.cell(0, 10, f'Page {self.page_no() - 1}', 0, 0, 'C')
        
        def title_page(self):
            """Create a professional title page"""
            self.add_page()
            
            # Add some space from top
            self.ln(40)
            
            # Main title
            self.set_font('Arial', 'B', 24)
            self.set_text_color(0, 51, 102)  # Dark blue
            self.cell(0, 15, 'EPIDEMIOLOGICAL REPORT', 0, 1, 'C')
            
            # Subtitle
            self.set_font('Arial', 'B', 16)
            self.set_text_color(0, 102, 153)  # Medium blue
            self.cell(0, 10, 'Comprehensive Analysis of Healthcare Data', 0, 1, 'C')
            
            # Add decorative line
            self.ln(20)
            self.set_draw_color(0, 102, 153)
            self.set_line_width(1)
            self.line(50, self.get_y(), 160, self.get_y())
            
            # Date and additional info
            self.ln(30)
            self.set_font('Arial', '', 12)
            self.set_text_color(80, 80, 80)
            
            current_date = datetime.now().strftime("%B %d, %Y")
            self.cell(0, 10, f'Report Generated: {current_date}', 0, 1, 'C')
            
            self.ln(10)
            self.cell(0, 10, 'Public Health Analytics Division', 0, 1, 'C')
            
            # Reset colors for content
            self.set_text_color(0, 0, 0)
        
        def chapter_title(self, title):
            """Add a formatted chapter title"""
            self.ln(10)
            self.set_font('Arial', 'B', 16)
            self.set_text_color(0, 51, 102)
            
            # Add background color for title
            self.set_fill_color(240, 248, 255)  # Light blue background
            self.cell(0, 12, title, 0, 1, 'L', True)
            self.ln(5)
            
            # Reset color
            self.set_text_color(0, 0, 0)
        
        def section_title(self, title):
            """Add a formatted section title"""
            self.ln(8)
            self.set_font('Arial', 'B', 14)
            self.set_text_color(0, 102, 153)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(3)
            
            # Reset color
            self.set_text_color(0, 0, 0)
        
        def subsection_title(self, title):
            """Add a formatted subsection title"""
            self.ln(5)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(51, 51, 51)
            self.cell(0, 8, title, 0, 1, 'L')
            self.ln(2)
            
            # Reset color
            self.set_text_color(0, 0, 0)
        
        def add_body_text(self, text):
            """Add formatted body text"""
            self.set_font('Arial', '', 11)
            self.set_text_color(40, 40, 40)
            
            # Clean up text and handle line breaks
            text = text.strip()
            if text:
                self.multi_cell(0, 6, text)
                self.ln(3)
    
    # Create PDF instance
    pdf = HealthReportPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    
    # Create title page
    pdf.title_page()
    
    # Start content on new page
    pdf.add_page()
    
    # Parse and format the report content
    lines = report_content.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect different heading levels and format accordingly
        if '**Executive Summary:**' in line:
            pdf.chapter_title("EXECUTIVE SUMMARY")
            current_section = "executive"
        elif '**Critical Risk Factors Identified:**' in line:
            pdf.chapter_title("CRITICAL RISK FACTORS IDENTIFIED")
            current_section = "risk_factors"
        elif '**Demographic Analysis & Vulnerable Populations:**' in line:
            pdf.chapter_title("DEMOGRAPHIC ANALYSIS & VULNERABLE POPULATIONS")
            current_section = "demographics"
        elif '**Geographic Distribution & Hotspots:**' in line:
            pdf.chapter_title("GEOGRAPHIC DISTRIBUTION & HOTSPOTS")
            current_section = "geographic"
        elif '**Temporal Trends & Seasonality:**' in line:
            pdf.chapter_title("TEMPORAL TRENDS & SEASONALITY")
            current_section = "temporal"
        elif '**Disease Progression Patterns:**' in line:
            pdf.chapter_title("DISEASE PROGRESSION PATTERNS")
            current_section = "disease_progression"
        elif '**Intervention Effectiveness Assessment:**' in line:
            pdf.chapter_title("INTERVENTION EFFECTIVENESS ASSESSMENT")
            current_section = "intervention"
        elif '**Strategic Recommendations:**' in line:
            pdf.chapter_title("STRATEGIC RECOMMENDATIONS")
            current_section = "recommendations"
        elif '**Immediate Action Items:**' in line:
            pdf.chapter_title("IMMEDIATE ACTION ITEMS")
            current_section = "action_items"
        elif '**Conclusion:**' in line:
            pdf.chapter_title("CONCLUSION")
            current_section = "conclusion"
        elif re.match(r'^\d+\.\s+\*\*.*?\*\*', line):
            # Numbered subsections with bold formatting (e.g., "1. **Compliance rate:**")
            # Extract the title and content
            match = re.match(r'^(\d+)\.\s+\*\*(.*?)\*\*:?\s*(.*)', line)
            if match:
                number, title, content = match.groups()
                pdf.subsection_title(f"{number}. {title.strip()}")
                if content.strip():
                    pdf.add_body_text(content.strip())
        elif line.startswith(('1. ', '2. ', '3. ', '4. ')) and '**' not in line:
            # Regular numbered points without bold formatting
            pdf.add_body_text(line)
        elif line.startswith('**') and line.endswith('**') and ':' not in line:
            # Section headers that we might have missed
            title = line.replace('**', '').strip()
            if title and len(title) < 100:  # Avoid treating long text as titles
                pdf.section_title(title)
        else:
            # Regular body text - clean up formatting
            cleaned_line = line.replace('**', '').strip()
            if cleaned_line and not cleaned_line.startswith('*') and len(cleaned_line) > 5:
                pdf.add_body_text(cleaned_line)
    
    # Save the PDF
    pdf.output(filename)
    print(f"Professional report saved as {filename}")
    print(f"Total pages: {pdf.page_no()}")

# Example usage
save_report_as_pdf(AIReport['report'], 'health_report.pdf')