import os
import sys
import subprocess
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
import logging
import traceback
from langchain.prompts import PromptTemplate
import json

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatNVIDIA(
    model_name="nvidia/llama-3.1-405b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Import data analysis functions
from data_analysis import *

# Create a comprehensive prompt template for health data analysis
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
    
    Generate a comprehensive, professional public health report based on this data.
    Aim for ~1500-2000 words total. If any critical fields are missing from the overview (e.g., no `timestamp`), please note these gaps and their potential impact on the analysis.
    """
)

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

global df 
df = load_existing_data()
df.to_json('all_data.json', orient='records', date_format='iso', indent=2)

# ------------------------ Flask App ------------------------ #
app = Flask(__name__)

# ------------------------ Real-Time Producer/Consumer ------------------------ #
producer_process = None
consumer_process = None
producer_output = []
consumer_output = []

def read_output(process, output_list, process_name):
    """Capture subprocess output and store last 50 lines."""
    try:
        for line in iter(process.stdout.readline, b''):
            if line:
                timestamp = datetime.now().strftime("%H:%M:%S")
                output_list.append(f"[{timestamp}] {line.decode('utf-8').strip()}")
                if len(output_list) > 50:
                    output_list.pop(0)
    except Exception as e:
        output_list.append(f"Error reading {process_name} output: {str(e)}")

# ------------------------ UI Routes ------------------------ #
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/refresh_DB', methods=['GET', 'POST'])
def refresh_DB():
    df = load_existing_data()
    return jsonify({"status": "Database refreshed successfully"})

@app.route('/data_analysis')
def data_analysis():
    return render_template('data_analysis.html')

def initialize_graphrag():
    """Initialize GraphRAG components"""
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"), 
            username="neo4j", 
            password=os.getenv("NEO4J_PASSWORD")
        )

        llm_graph = ChatGroq(
            groq_api_key=os.environ.get('GROQ_API'),
            model_name='llama-3.3-70b-versatile',
            max_tokens=2048,
        )

        CYPHER_GENERATION_TEMPLATE_XML = """<cypher_generation_prompt>
            <instructions>
                <title>Instructions for Neo4j Cypher Query Generation</title>
                <rule>You are a world-class Neo4j Cypher query translator. Your sole purpose is to convert a user's question into a valid and efficient Cypher query based on the provided graph schema.</rule>
                <rule>Strictly adhere to the schema. Never use node labels, relationship types, or property names that are not explicitly defined in the `<schema>` block.</rule>
                <rule>Use underscores in multi-word names like 'coastal_region'.</rule>
                <rule>Your output MUST be a single, valid Cypher query and nothing else. No explanations, no markdown, no assumptions.</rule>
                <rule>Do not guess. If the query cannot be generated without assuming missing information, return an empty query.</rule>
                <rule>You are allowed to match symptoms to cases and infer probable diseases based on frequency counts of co-occurrence.</rule>
                <rule>Use aggregation such as count(*) and ORDER BY when needed to find most common patterns.</rule>
                <rule>Whenever asked for age average you need to return an integer.</rule>
                <rule>Avoid nested aggregates (e.g. `avg(avg(...))`).</rule>
                <rule>Do NOT use `collect(...)` followed by `avg(...)` on the collected list—you must aggregate directly on the numeric property in a single pass.</rule>
                <rule>If the question asks for 'most effective', include avg(i.effectivenessScore) and sort by it descending.</rule>
                <rule>Pay careful attention to relationship directions in the schema.</rule>
                <rule>Whenever asked for symptoms, you should only return the symptom names, not the full text descriptions and give common symptoms everytime, do not give all symptoms of every disease as it would be very long for context.</rule>
                <rule>Fix the token limit to be inbetween 5000 and 9000 tokens, so that the model can handle complex queries without truncation.</rule>
                <rule>When asked for symptoms, only make query for most common symptoms, not all symptoms.</rule>
                <rule>When asked for intervention name you should return the intervention type, not the intervention id.</rule>
                <rule>When using ORDER BY on an aggregate like count(), avg(), etc., ensure the aggregate is aliased in the RETURN or WITH clause before ordering—e.g., `RETURN x AS y, count(c) AS frequency ORDER BY frequency DESC`.</rule>
            </instructions>

            <schema>
                {schema}
            </schema>
            
            <few_shot_examples>
                <example>
                    <question>Which intervention type was most effective for reducing severe malaria cases in the coastal region during the last two years, and what was the average age of patients who benefited from these interventions?</question>
                    <cypher>
        MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease {{name: 'malaria'}}),
            (c)-[:OCCURRED_IN]->(r:Region {{name: 'coastal_region'}}),
            (c)-[:AFFECTED_BY]->(i:Intervention),
            (p:Patient)-[:REPORTED]->(c),
            (c)-[:HAS_SEVERITY]->(s:SeverityLevel {{level: 'severe'}}),
            (c)-[:REPORTED_IN_YEAR]->(y:Year)
        WHERE y.year >= date().year - 2
        RETURN i.type AS intervention_type, 
            toInteger(avg(p.age)) AS average_patient_age,
            avg(i.effectivenessScore) AS avg_effectiveness
        ORDER BY avg_effectiveness DESC
        LIMIT 1
                    </cypher>
                </example>

                <example>
                    <question>What is the total number of malaria cases reported in the coastal region last year?</question>
                    <cypher>
        MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease {{name: 'malaria'}}),
            (c)-[:OCCURRED_IN]->(r:Region {{name: 'coastal_region'}}),
            (c)-[:REPORTED_IN_YEAR]->(y:Year)
        WHERE y.year = date().year - 1
        RETURN count(c) AS total_cases
                    </cypher>
                </example>

                <example>
                    <question>Which disease has the highest average age of patients in urban areas?</question>
                    <cypher>
        MATCH (p:Patient)-[:REPORTED]->(c:Case)-[:DIAGNOSED_WITH]->(d:Disease),
            (c)-[:OCCURRED_IN]->(r:Region {{name: 'urban_area'}})
        RETURN d.name AS disease_name, 
            toInteger(avg(p.age)) AS average_age
        ORDER BY average_age DESC
        LIMIT 1
                    </cypher>
                </example>

                <example>
                    <question>What are the most common symptoms for dengue fever cases?</question>
                    <cypher>
        MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease {{name: 'dengue_fever'}}),
            (c)-[:PRESENTED_SYMPTOM]->(s:Symptom)
        RETURN s.name AS symptom, count(c) AS frequency
        ORDER BY frequency DESC
        LIMIT 5
                    </cypher>
                </example>

                <example>
                <question>Which intervention type had the highest effectiveness score in reducing severe cases across different disease categories, and what was the average patient age and most common symptoms for those cases in the last two years?</question>
                <cypher>
                MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease),
                    (c)-[:AFFECTED_BY]->(i:Intervention),
                    (p:Patient)-[:REPORTED]->(c),
                    (c)-[:PRESENTED_SYMPTOM]->(s:Symptom),
                    (c)-[:HAS_SEVERITY]->(sev:SeverityLevel {{level: 'severe'}}),
                    (d)-[:IS_A_TYPE_OF]->(dc:DiseaseCategory),
                    (c)-[:REPORTED_IN_YEAR]->(y:Year)
                WHERE y.year >= date().year - 2
                WITH i.type AS intervention_type, 
                    avg(i.effectivenessScore) AS avg_effectiveness,
                    collect(p.age) AS patient_ages,
                    collect(s.name) AS symptoms,
                    count(c) AS case_count
                RETURN intervention_type, 
                    toInteger(avg([age IN patient_ages | age])) AS average_patient_age,
                    avg_effectiveness,
                    [symptom IN symptoms | symptom][0..3] AS common_symptoms,
                    case_count
                ORDER BY avg_effectiveness DESC
                LIMIT 1
                </cypher>
                </example>

                <example>
                    <question>Which region had the most severe tuberculosis cases in the last 6 months?</question>
                    <cypher>
        MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease {{name: 'tuberculosis'}}),
            (c)-[:OCCURRED_IN]->(r:Region),
            (c)-[:HAS_SEVERITY]->(s:SeverityLevel {{level: 'severe'}})
        WHERE c.timestamp >= datetime() - duration({{months: 6}})
        RETURN r.name AS region_name, count(c) AS severe_cases
        ORDER BY severe_cases DESC
        LIMIT 1
                    </cypher>
                </example>

                <example>
                    <question>What is the average effectiveness score of interventions targeting malaria?</question>
                    <cypher>
        MATCH (i:Intervention)-[:TARGETS_DISEASE]->(d:Disease {{name: 'malaria'}})
        RETURN avg(i.effectivenessScore) AS average_effectiveness
                    </cypher>
                </example>

                <example>
                    <question>Which gender has more cases of severe diseases in the mountain region?</question>
                    <cypher>
        MATCH (p:Patient)-[:REPORTED]->(c:Case),
            (p)-[:HAS_GENDER]->(g:Gender),
            (c)-[:OCCURRED_IN]->(r:Region {{name: 'mountain_region'}}),
            (c)-[:HAS_SEVERITY]->(s:SeverityLevel {{level: 'severe'}})
        RETURN g.value AS gender, count(c) AS case_count
        ORDER BY case_count DESC
        LIMIT 1
                    </cypher>
                </example>

                <example>
                    <question>Find patients who have a history of malaria and currently live in the coastal region</question>
                    <cypher>
        MATCH (p:Patient)-[:HAS_HISTORY_OF]->(d:Disease {{name: 'malaria'}}),
            (p)-[:LIVES_IN]->(r:Region {{name: 'coastal_region'}})
        RETURN p.patientId, p.age, p.gender
                    </cypher>
                </example>
            </few_shot_examples>
            
            <important_notes>
                <note>Symptom names in the database are long text strings like "Symptoms include fever, cough and chills".</note>
                <note>To match individual symptoms, use `CONTAINS` or `=~` instead of `s.name IN [...]`.</note>
                <note>Remember relationship directions: Patient-[:REPORTED]->Case, Case-[:DIAGNOSED_WITH]->Disease</note>
                <note>Use date() functions for current date comparisons</note>
                <note>Use datetime() for timestamp comparisons with duration</note>
                <note>Always use toInteger() when returning average ages</note>
            </important_notes>

            <common_patterns>
                <pattern name="Most Effective Intervention">
                    <description>When asking for most effective intervention, always include effectiveness score in ordering</description>
                    <template>
        MATCH (relevant patterns)
        RETURN intervention_details, avg(i.effectivenessScore) AS effectiveness
        ORDER BY effectiveness DESC
        LIMIT 1
                    </template>
                </pattern>

                <pattern name="Age Aggregation">
                    <description>Direct aggregation on patient age property</description>
                    <template>
        MATCH (p:Patient)-[:REPORTED]->(c:Case)
        WHERE (conditions)
        RETURN toInteger(avg(p.age)) AS average_age
                    </template>
                </pattern>

                <pattern name="Time-based Filtering">
                    <description>Filtering by years or recent time periods</description>
                    <template>
        MATCH (c:Case)-[:REPORTED_IN_YEAR]->(y:Year)
        WHERE y.year >= date().year - 2
                    </template>
                </pattern>

                <pattern name="Disease-Region-Severity">
                    <description>Common pattern for disease analysis by region and severity</description>
                    <template>
        MATCH (c:Case)-[:DIAGNOSED_WITH]->(d:Disease {{name: 'disease_name'}}),
            (c)-[:OCCURRED_IN]->(r:Region {{name: 'region_name'}}),
            (c)-[:HAS_SEVERITY]->(s:SeverityLevel {{level: 'severity_level'}})
                    </template>
                </pattern>
            </common_patterns>

            <task>
                <title>Current Task</title>
                <question>{question}</question>
                <cypher>
                </cypher>
            </task>
        </cypher_generation_prompt>"""

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE_XML
        )

        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm_graph,
            verbose=True, 
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True
        )

        return chain
    
    except Exception as e:
        logging.error(f"Failed to initialize GraphRAG: {str(e)}")
        return None

# Initialize GraphRAG chain at startup
graphrag_chain = initialize_graphrag()

@app.route('/graphrag_chatbot')
def graphrag_chatbot():
    """Render the GraphRAG chatbot page"""
    return render_template('graphrag_chatbot.html')

@app.route('/api/graphrag/chat', methods=['POST'])
def graphrag_chat():
    """Handle GraphRAG chat requests"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        # Check if GraphRAG is initialized
        if not graphrag_chain:
            return jsonify({
                'success': False,
                'error': 'GraphRAG system is not available. Please check your Neo4j connection and API keys.'
            }), 503
        
        # Process the query through GraphRAG
        try:
            result = graphrag_chain.invoke({"query": user_message})
            
            # Extract the answer from the result
            if isinstance(result, dict) and 'result' in result:
                answer = result['result']
            else:
                answer = str(result)
            
            # Format the response for better readability
            if not answer or answer.strip() == "":
                answer = "I couldn't find a specific answer to your question in the knowledge graph. Please try rephrasing your question or ask about epidemiological data patterns, disease cases, interventions, or regional health statistics."
            
            return jsonify({
                'success': True,
                'response': answer,
                'query': user_message
            })
            
        except Exception as query_error:
            logging.error(f"GraphRAG query error: {str(query_error)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Provide a more user-friendly error message
            error_msg = "I encountered an issue processing your question. "
            
            if "syntax error" in str(query_error).lower():
                error_msg += "Please try rephrasing your question in simpler terms."
            elif "connection" in str(query_error).lower():
                error_msg += "There seems to be a database connection issue. Please try again later."
            elif "timeout" in str(query_error).lower():
                error_msg += "Your query is taking too long to process. Please try a simpler question."
            else:
                error_msg += "Please try asking about specific diseases, regions, interventions, or patient statistics."
            
            return jsonify({
                'success': True,  # We still return success to show the error message to user
                'response': error_msg,
                'query': user_message
            })
    
    except Exception as e:
        logging.error(f"GraphRAG chat API error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': 'Internal server error occurred while processing your request.'
        }), 500

@app.route('/api/graphrag/health', methods=['GET'])
def graphrag_health():
    """Check GraphRAG system health"""
    try:
        if not graphrag_chain:
            return jsonify({
                'status': 'unhealthy',
                'message': 'GraphRAG system not initialized'
            }), 503
        
        # Try a simple test query
        test_result = graphrag_chain.invoke({"query": "What is the schema of this database?"})
        
        return jsonify({
            'status': 'healthy',
            'message': 'GraphRAG system is operational'
        })
        
    except Exception as e:
        logging.error(f"GraphRAG health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': f'System check failed: {str(e)}'
        }), 503

@app.route('/api/graphrag/suggestions', methods=['GET'])
def graphrag_suggestions():
    """Get quick query suggestions for users"""
    suggestions = [
        "What are the most common symptoms of malaria?",
        "Which region has the highest number of severe cases?",
        "What is the average age of patients with tuberculosis?",
        "Which intervention type is most effective for dengue fever?",
        "Show me disease patterns in coastal regions",
        "What are the recent trends in vaccination effectiveness?",
        "Which diseases are most common in urban areas?",
        "What is the gender distribution of severe cases?"
    ]
    
    return jsonify({
        'success': True,
        'suggestions': suggestions
    })

@app.route('/ai_agent_simulation')
def ai_agent_simulation():
    return render_template('ai_agent_simulation.html')

# ------------------------ Data Analysis Report Generation ------------------------ #
@app.route('/api/generate_report', methods=['POST'])
def generate_ai_report():
    """Generate AI-powered health report using NVIDIA LLM"""
    try:
        # Get request parameters
        data = request.get_json()
        analysis_focus = data.get('focus', 'comprehensive_analysis')
        report_type = data.get('type', 'epidemiological_report')
        
        # Access the global dataframe
        global df
        if df is None or df.empty:
            return jsonify({
                "error": "No data available. Please ensure data is loaded.",
                "status": "error"
            }), 400
        
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
        print("Prompt sent to LLM:")
        print(prompt)
        
        response = llm.invoke(prompt)
        
        print("LLM response received:")  # Debug log
        print(response)
        
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
        
        return jsonify(report_data)
    
    except Exception as e:
        print(f"Error generating report: {str(e)}")  # Debug log
        return jsonify({
            "error": f"Failed to generate report: {str(e)}",
            "status": "error"
        }), 500
    
@app.route('/api/analyze_specific_pattern', methods=['POST'])
def analyze_specific_pattern():
    """Analyze specific patterns in the data based on user query"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        global df
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 400
        
        # Custom analysis prompt
        analysis_prompt = PromptTemplate(
            input_variables=["query", "data_info"],
            template="""
            As a data scientist specializing in health analytics, answer this specific question about the dataset:
            
            USER QUERY: {query}
            
            AVAILABLE DATA:
            {data_info}
            
            Provide a detailed analysis addressing the user's question. Include:
            1. Direct answer to the question
            2. Supporting evidence from the data
            3. Implications for public health
            4. Recommendations based on findings
            
            Be specific and cite relevant data points.
            """
        )
        
        # Prepare relevant data info
        data_info = {
            "columns": list(df.columns),
            "sample_size": len(df),
            "key_stats": prepare_dataframe_summary(df)
        }
        
        prompt = analysis_prompt.format(
            query=user_query,
            data_info=json.dumps(data_info, indent=2, default=str)
        )
        
        response = llm.invoke(prompt)
        
        return jsonify({
            "analysis": response.content if hasattr(response, 'content') else str(response),
            "query": user_query,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

# ------------------------ Simulation Control ------------------------ #
@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    global producer_process, consumer_process, producer_output, consumer_output

    try:
        producer_output.clear()
        consumer_output.clear()

        base_path = os.path.join(os.getcwd(), "dataset")
        producer_path = os.path.join(base_path, "producer.py")
        consumer_path = os.path.join(base_path, "consumer.py")

        producer_process = subprocess.Popen(
            [sys.executable, "-u", producer_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=False
        )

        consumer_process = subprocess.Popen(
            [sys.executable, "-u", consumer_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=False
        )

        threading.Thread(target=read_output, args=(producer_process, producer_output, "Producer"), daemon=True).start()
        threading.Thread(target=read_output, args=(consumer_process, consumer_output, "Consumer"), daemon=True).start()

        return jsonify({'status': 'success', 'message': 'Simulation started successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error starting simulation: {str(e)}'})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    global producer_process, consumer_process

    try:
        if producer_process:
            producer_process.terminate()
            producer_process = None
        if consumer_process:
            consumer_process.terminate()
            consumer_process = None

        return jsonify({'status': 'success', 'message': 'Simulation stopped successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error stopping simulation: {str(e)}'})

@app.route('/get_output')
def get_output():
    return jsonify({
        'producer_output': producer_output[-20:],
        'consumer_output': consumer_output[-20:]
    })

# ------------------------ Data Analysis API Routes ---------------------- #
@app.route('/api/outbreaks')
def api_outbreaks():
    return jsonify(analyze_outbreak_patterns(df))

@app.route('/api/demographic_risk')
def api_demographic_risk():
    return jsonify(analyze_demographic_risk_factors(df))

@app.route('/api/intervention_effectiveness')
def api_intervention_effectiveness():
    return jsonify(analyze_intervention_effectiveness(df))

@app.route('/api/geographic_spread')
def api_geographic_spread():
    return jsonify(analyze_geographic_spread_patterns(df))

@app.route('/api/temporal_disease_evolution')
def api_temporal_disease_evolution():
    return jsonify(analyze_temporal_disease_evolution(df))

# ------------------------ Run Flask ------------------------ #
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
