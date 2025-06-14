import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    
    def split_temporal_data(self, df, split_date=None):
        """Split data into baseline (older) and current (recent) periods"""
        if split_date is None:
            # Use median timestamp as split point
            split_date = df['timestamp'].median()
        
        baseline = df[df['timestamp'] <= split_date].copy()
        current = df[df['timestamp'] > split_date].copy()
        
        print(f"Split date: {split_date}")
        print(f"Baseline period: {len(baseline)} records")
        print(f"Current period: {len(current)} records")
        
        return baseline, current
    
    def calculate_distribution_drift(self, baseline_col, current_col, column_name):
        """Calculate drift metrics for a specific column"""
        drift_results = {
            'column': column_name,
            'baseline_mean': None,
            'current_mean': None,
            'mean_change': None,
            'ks_statistic': None,
            'ks_p_value': None,
            'drift_detected': False
        }
        
        try:
            # Remove NaN values
            baseline_clean = baseline_col.dropna()
            current_clean = current_col.dropna()
            
            if len(baseline_clean) == 0 or len(current_clean) == 0:
                return drift_results
            
            # For numerical columns
            if baseline_clean.dtype in ['int64', 'float64']:
                drift_results['baseline_mean'] = baseline_clean.mean()
                drift_results['current_mean'] = current_clean.mean()
                drift_results['mean_change'] = (drift_results['current_mean'] - drift_results['baseline_mean']) / drift_results['baseline_mean'] * 100
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(baseline_clean, current_clean)
                drift_results['ks_statistic'] = ks_stat
                drift_results['ks_p_value'] = ks_p
                drift_results['drift_detected'] = ks_p < 0.05
            
            # For categorical columns
            else:
                baseline_dist = baseline_clean.value_counts(normalize=True)
                current_dist = current_clean.value_counts(normalize=True)
                
                # Chi-square test for categorical drift
                all_categories = set(baseline_dist.index) | set(current_dist.index)
                baseline_counts = [baseline_dist.get(cat, 0) * len(baseline_clean) for cat in all_categories]
                current_counts = [current_dist.get(cat, 0) * len(current_clean) for cat in all_categories]
                
                if len(all_categories) > 1:
                    chi2_stat, chi2_p = stats.chisquare(current_counts, baseline_counts)
                    drift_results['ks_statistic'] = chi2_stat
                    drift_results['ks_p_value'] = chi2_p
                    drift_results['drift_detected'] = chi2_p < 0.05
        
        except Exception as e:
            print(f"Error calculating drift for {column_name}: {e}")
        
        return drift_results
    
    def analyze_data_drift(self):
        """Main function to analyze data drift"""
        print("Extracting data from Neo4j...")
        df = self.extract_all_data()
        
        if df.empty:
            print("No data found in Neo4j!")
            return None
        
        print(f"Total records extracted: {len(df)}")
        
        # Split data into baseline and current periods
        self.baseline_data, self.current_data = self.split_temporal_data(df)
        
        # Columns to analyze for drift
        numerical_cols = ['age', 'latitude', 'longitude', 'intervention_effectiveness', 
                         'intervention_cost', 'compliance_rate', 'symptoms_count']
        categorical_cols = ['disease', 'region', 'severity', 'gender', 'age_group', 
                           'intervention_type', 'is_outbreak_related', 'contact_tracing_needed',
                           'hospitalization_required']
        
        all_cols = numerical_cols + categorical_cols
        drift_results = []
        
        print("\nAnalyzing drift for each column...")
        for col in all_cols:
            if col in df.columns:
                result = self.calculate_distribution_drift(
                    self.baseline_data[col], 
                    self.current_data[col], 
                    col
                )
                drift_results.append(result)
        
        drift_df = pd.DataFrame(drift_results)
        return drift_df
    
    def generate_drift_report(self, drift_df):
        """Generate a comprehensive drift report"""
        if drift_df is None:
            return
        
        print("\n" + "="*60)
        print("DATA DRIFT ANALYSIS REPORT")
        print("="*60)
        
        # Summary statistics
        total_columns = len(drift_df)
        drifted_columns = len(drift_df[drift_df['drift_detected'] == True])
        
        print(f"\nSUMMARY:")
        print(f"Total columns analyzed: {total_columns}")
        print(f"Columns with drift detected: {drifted_columns}")
        print(f"Drift rate: {(drifted_columns/total_columns)*100:.1f}%")
        
        # Detailed drift results
        print(f"\nDETAILED RESULTS:")
        print("-"*60)
        
        for _, row in drift_df.iterrows():
            status = "🚨 DRIFT DETECTED" if row['drift_detected'] else "✅ No Drift"
            print(f"\nColumn: {row['column']}")
            print(f"Status: {status}")
            
            if row['baseline_mean'] is not None:
                print(f"Baseline Mean: {row['baseline_mean']:.3f}")
                print(f"Current Mean: {row['current_mean']:.3f}")
                print(f"Mean Change: {row['mean_change']:.2f}%")
            
            if row['ks_p_value'] is not None:
                print(f"Statistical Test p-value: {row['ks_p_value']:.6f}")
                print(f"Test Statistic: {row['ks_statistic']:.6f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-"*60)
        
        if drifted_columns == 0:
            print("✅ No significant drift detected. Data distribution remains stable.")
        else:
            drifted_cols = drift_df[drift_df['drift_detected'] == True]['column'].tolist()
            print(f"⚠️  Monitor the following columns with detected drift: {', '.join(drifted_cols)}")
            print("   Consider retraining models or updating data processing pipelines.")
            
            # Specific recommendations based on drift type
            high_drift_cols = drift_df[
                (drift_df['drift_detected'] == True) & 
                (drift_df['mean_change'].abs() > 20)
            ]['column'].tolist()
            
            if high_drift_cols:
                print(f"🔥 High drift (>20% change) detected in: {', '.join(high_drift_cols)}")
                print("   Immediate attention required!")
    
    def create_visualizations(self, drift_df):
        """Create improved visualizations for drift analysis with 3 focused graphs"""
        if drift_df is None or self.baseline_data is None or self.current_data is None:
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Data Drift Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Drift Severity Heatmap - Shows which columns have drift and their severity
        drift_matrix = drift_df.copy()
        drift_matrix['drift_severity'] = 0
        
        # Calculate drift severity based on p-value and mean change
        for idx, row in drift_matrix.iterrows():
            if row['drift_detected']:
                # Use p-value to determine severity (lower p-value = higher severity)
                p_val_score = max(0, 1 - row['ks_p_value']) if pd.notna(row['ks_p_value']) else 0
                
                # Use mean change magnitude for numerical columns
                mean_change_score = 0
                if pd.notna(row['mean_change']):
                    mean_change_score = min(abs(row['mean_change']) / 100, 1)  # Normalize to 0-1
                
                # Combined severity score
                drift_matrix.loc[idx, 'drift_severity'] = max(p_val_score, mean_change_score)
        
        # Create heatmap data
        heatmap_data = drift_matrix[['column', 'drift_severity']].set_index('column')
        heatmap_data = heatmap_data.sort_values('drift_severity', ascending=False)
        
        # Plot heatmap
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='Reds', 
                    cbar_kws={'label': 'Drift Severity'}, ax=axes[0])
        axes[0].set_title('Drift Severity by Column', fontweight='bold', pad=20)
        axes[0].set_xlabel('Columns', fontweight='bold')
        axes[0].set_ylabel('')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Statistical Significance vs Effect Size Scatter Plot
        numerical_drift = drift_df[drift_df['mean_change'].notna()].copy()
        
        if not numerical_drift.empty:
            # Convert p-values to -log10 scale for better visualization
            numerical_drift['neg_log_pval'] = -np.log10(numerical_drift['ks_p_value'].clip(lower=1e-10))
            
            # Create scatter plot
            colors = ['red' if drift else 'blue' for drift in numerical_drift['drift_detected']]
            scatter = axes[1].scatter(numerical_drift['mean_change'], numerical_drift['neg_log_pval'], 
                                    c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            
            # Add significance threshold line
            axes[1].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                        label='Significance Threshold (p=0.05)')
            
            # Add effect size threshold lines
            axes[1].axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='High Effect Size (±20%)')
            axes[1].axvline(x=-20, color='orange', linestyle='--', alpha=0.7)
            
            # Annotate points with column names
            for idx, row in numerical_drift.iterrows():
                if row['drift_detected'] or abs(row['mean_change']) > 15:
                    axes[1].annotate(row['column'], 
                                (row['mean_change'], row['neg_log_pval']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.8)
            
            axes[1].set_xlabel('Mean Change (%)', fontweight='bold')
            axes[1].set_ylabel('-log₁₀(p-value)', fontweight='bold')
            axes[1].set_title('Statistical Significance vs Effect Size', fontweight='bold', pad=20)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Create custom legend for colors
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='Drift Detected'),
                            Patch(facecolor='blue', label='No Drift')]
            axes[1].legend(handles=legend_elements, loc='upper right')
        
        # 3. Distribution Comparison for Top Drifted Column
        top_drift_col = drift_df[drift_df['drift_detected'] == True]
        
        if not top_drift_col.empty:
            # Get the column with highest drift severity
            if 'mean_change' in top_drift_col.columns:
                top_col = top_drift_col.loc[top_drift_col['mean_change'].abs().idxmax(), 'column']
            else:
                top_col = top_drift_col.iloc[0]['column']
            
            # Plot distributions
            baseline_data = self.baseline_data[top_col].dropna()
            current_data = self.current_data[top_col].dropna()
            
            if baseline_data.dtype in ['int64', 'float64']:
                # Numerical data - use histograms
                axes[2].hist(baseline_data, alpha=0.6, label='Baseline Period', 
                            bins=20, color='blue', density=True)
                axes[2].hist(current_data, alpha=0.6, label='Current Period', 
                            bins=20, color='red', density=True)
                axes[2].set_ylabel('Density', fontweight='bold')
            else:
                # Categorical data - use bar plots
                baseline_counts = baseline_data.value_counts(normalize=True)
                current_counts = current_data.value_counts(normalize=True)
                
                x = np.arange(len(baseline_counts))
                width = 0.35
                
                axes[2].bar(x - width/2, baseline_counts.values, width, 
                        label='Baseline Period', alpha=0.7, color='blue')
                axes[2].bar(x + width/2, current_counts.values, width, 
                        label='Current Period', alpha=0.7, color='red')
                
                axes[2].set_xticks(x)
                axes[2].set_xticklabels(baseline_counts.index, rotation=45)
                axes[2].set_ylabel('Proportion', fontweight='bold')
            
            axes[2].set_xlabel(f'{top_col}', fontweight='bold')
            axes[2].set_title(f'Distribution Comparison: {top_col}', fontweight='bold', pad=20)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        else:
            # If no drift detected, show overall case distribution over time
            timeline_data = pd.concat([
                self.baseline_data[['timestamp']].assign(period='Baseline'),
                self.current_data[['timestamp']].assign(period='Current')
            ])
            
            # Create weekly aggregation
            timeline_data['week'] = timeline_data['timestamp'].dt.to_period('W')
            weekly_counts = timeline_data.groupby(['week', 'period']).size().unstack(fill_value=0)
            
            if not weekly_counts.empty:
                weekly_counts.plot(kind='bar', ax=axes[2], alpha=0.7, width=0.8)
                axes[2].set_xlabel('Week', fontweight='bold')
                axes[2].set_ylabel('Number of Cases', fontweight='bold')
                axes[2].set_title('Weekly Case Distribution', fontweight='bold', pad=20)
                axes[2].tick_params(axis='x', rotation=45)
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_data_drift_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Improved visualization saved as 'improved_data_drift_analysis.png'")
        
        # Print insights
        print(f"\n🔍 VISUALIZATION INSIGHTS:")
        print("-" * 50)
        
        high_severity_cols = drift_df[drift_df['drift_detected'] == True]
        if not high_severity_cols.empty:
            print(f"📈 Columns with detected drift: {len(high_severity_cols)}")
            for _, row in high_severity_cols.iterrows():
                severity = "HIGH" if (pd.notna(row['mean_change']) and abs(row['mean_change']) > 20) else "MODERATE"
                print(f"   • {row['column']}: {severity} severity drift")
        else:
            print("✅ No significant drift patterns detected across all columns")

def main():
    analyzer = Neo4jDataDriftAnalyzer()
    
    try:
        # Perform drift analysis
        drift_results = analyzer.analyze_data_drift()
        
        if drift_results is not None:
            # Generate report
            analyzer.generate_drift_report(drift_results)
            
            # Create visualizations
            analyzer.create_visualizations(drift_results)
            
            # Save results to CSV
            drift_results.to_csv('drift_analysis_results.csv', index=False)
            print(f"\n💾 Results saved to 'drift_analysis_results.csv'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    finally:
        analyzer.close()

if __name__ == '__main__':
    main()