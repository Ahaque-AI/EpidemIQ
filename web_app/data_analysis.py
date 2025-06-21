import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from warnings import filterwarnings

filterwarnings("ignore", category=FutureWarning)

def convert_to_serializable(obj):
    """Helper function to convert non-serializable types"""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, tuple):
        return str(obj)
    else:
        return obj

def analyze_outbreak_patterns(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(str)
    df['month'] = df['timestamp'].dt.month.astype(str)
    
    # Temporal clustering analysis
    daily_cases = df.groupby('date').size()
    rolling_avg = daily_cases.rolling(window=7).mean()
    outbreak_threshold = rolling_avg.mean() + (2 * rolling_avg.std()) if not rolling_avg.empty else 0
    outbreak_days = daily_cases[daily_cases > outbreak_threshold].index.tolist()
    
    result = {
        'outbreak_summary': {
            'total_outbreak_days': len(outbreak_days),
            'outbreak_threshold': round(float(outbreak_threshold), 2),
            'peak_cases_day': daily_cases.idxmax().strftime('%Y-%m-%d') if not daily_cases.empty else None,
            'peak_cases_count': int(daily_cases.max()) if not daily_cases.empty else 0
        },
        'temporal_trends': {
            'weekly_cases': df.groupby('week').size().to_dict(),
            'monthly_distribution': df.groupby('month').size().to_dict()
        }
    }
    
    return convert_to_serializable(result)

def analyze_demographic_risk_factors(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    
    total_cases = len(df)
    
    # Age-based risk stratification
    age_risk = df.groupby('age_group').agg({
        'case_id': 'count',
        'severity': lambda x: (x == 'severe').sum()
    })
    age_risk['severity_rate'] = (age_risk['severity'] / age_risk['case_id'] * 100).round(2)
    
    # Gender-based analysis
    gender_disease_matrix = pd.crosstab(df['gender'], df['disease'], normalize='index') * 100
    gender_severity = df.groupby('gender').agg({
        'severity': lambda x: (x == 'severe').sum() / len(x) * 100,
        'case_id': 'count'
    }).round(2)
    
    # Multi-dimensional risk scoring
    risk_factors = []
    for age in df['age_group'].unique():
        for gender in df['gender'].unique():
            subset = df[(df['age_group'] == age) & (df['gender'] == gender)]
            if len(subset) > 0:
                severe_rate = (subset['severity'] == 'severe').sum() / len(subset) * 100
                
                risk_factors.append({
                    'age_group': str(age),
                    'gender': str(gender),
                    'case_count': len(subset),
                    'severity_rate': round(severe_rate, 2)
                })
    
    risk_factors = sorted(risk_factors, key=lambda x: x['severity_rate'], reverse=True)
    
    result = {
        'age_risk_profiles': age_risk.to_dict('index'),
        'gender_disease_patterns': gender_disease_matrix.to_dict('index'),
        'gender_severity_analysis': gender_severity.to_dict('index'),
        'high_risk_demographics': risk_factors[:10]
    }
    
    return convert_to_serializable(result)

def analyze_intervention_effectiveness(df: pd.DataFrame) -> Dict:
    if df.empty or 'intervention_type' not in df.columns:
        return {}
    
    # Cost-effectiveness analysis
    intervention_analysis = df.groupby('intervention_type').agg({
        'intervention_effectiveness': ['mean', 'std', 'count'],
        'intervention_cost': ['mean', 'sum'],
        'compliance_rate': 'mean',
        'severity': lambda x: (x == 'severe').sum(),
        'case_id': 'count'
    }).round(2)
    
    intervention_analysis.columns = [
        'avg_effectiveness', 'effectiveness_std', 'sample_size',
        'avg_cost_per_case', 'total_cost', 'avg_compliance',
        'severe_cases', 'total_cases'
    ]
    
    result = {
        'intervention_performance': intervention_analysis.to_dict('index')
    }
    
    return convert_to_serializable(result)

def analyze_geographic_spread_patterns(df: pd.DataFrame) -> Dict:
    if df.empty or not all(col in df.columns for col in ['latitude', 'longitude']):
        return {}
    
    geo_df = df.dropna(subset=['latitude', 'longitude']).copy()
    if geo_df.empty:
        return {}
    
    # Basic geographic distribution
    region_counts = geo_df.groupby('region').size().sort_values(ascending=False).to_dict()
    
    # Severity distribution by region
    severity_by_region = geo_df.groupby('region')['severity'].apply(
        lambda x: (x == 'severe').sum() / len(x) * 100
    ).sort_values(ascending=False).to_dict()
    
    # Top regions by case count
    top_regions = list(region_counts.keys())[:5]
    
    # Basic risk assessment
    regional_risk = {}
    for region in region_counts:
        regional_risk[region] = {
            'case_count': region_counts[region],
            'severity_rate': severity_by_region.get(region, 0),
            'risk_level': 'High' if severity_by_region.get(region, 0) > 30 else 'Medium'
        }
    
    result = {
        'basic_geographic_analysis': {
            'total_unique_regions': len(region_counts),
            'largest_region': max(region_counts, key=region_counts.get) if region_counts else None,
            'smallest_region': min(region_counts, key=region_counts.get) if region_counts else None,
            'top_regions': top_regions
        },
        'regional_severity_rates': severity_by_region,
        'regional_case_counts': region_counts,
        'regional_risk_assessment': regional_risk
    }
    
    return convert_to_serializable(result)

def analyze_temporal_disease_evolution(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(str)
    df['month'] = df['timestamp'].dt.month.astype(str)
    
    # Disease lifecycle analysis
    disease_lifecycles = {}
    for disease in df['disease'].unique():
        disease_data = df[df['disease'] == disease].sort_values('timestamp')
        
        if len(disease_data) > 1:
            daily_cases = disease_data.groupby('date').size()
            peak_cases = daily_cases.max() if not daily_cases.empty else 0
            
            disease_lifecycles[disease] = {
                'total_cases': len(disease_data),
                'peak_cases': int(peak_cases),
                'avg_severity': (disease_data['severity'] == 'severe').mean() * 100
            }
    
    # Seasonal patterns
    seasonal_analysis = {
        'monthly_patterns': df.groupby('month').size().to_dict()
    }
    
    result = {
        'disease_lifecycle_analysis': disease_lifecycles,
        'seasonal_patterns': seasonal_analysis
    }
    
    return convert_to_serializable(result)

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