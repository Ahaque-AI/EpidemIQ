import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import json
from typing import List, Dict, Tuple
import uuid

# Initialize Faker for generating realistic data
fake = Faker()

class SyntheticHealthDataGenerator:
    """
    Comprehensive synthetic health data generator for public health monitoring
    Generates realistic health reports with temporal and geographical patterns
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Disease definitions with symptoms, severity, and outbreak patterns
        self.diseases = {
            'influenza': {
                'symptoms': ['fever', 'cough', 'headache', 'body_aches', 'fatigue', 'chills'],
                'severity_levels': ['mild', 'moderate', 'severe'],
                'seasonal_pattern': 'winter',  # Higher in winter months
                'contagious': True,
                'incubation_days': (1, 4),
                'duration_days': (5, 14),
                'outbreak_probability': 0.3
            },
            'covid19': {
                'symptoms': ['fever', 'dry_cough', 'shortness_of_breath', 'loss_of_taste', 'loss_of_smell', 'fatigue'],
                'severity_levels': ['mild', 'moderate', 'severe', 'critical'],
                'seasonal_pattern': None,
                'contagious': True,
                'incubation_days': (2, 14),
                'duration_days': (10, 21),
                'outbreak_probability': 0.4
            },
            'gastroenteritis': {
                'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever'],
                'severity_levels': ['mild', 'moderate', 'severe'],
                'seasonal_pattern': 'summer',
                'contagious': True,
                'incubation_days': (1, 3),
                'duration_days': (3, 7),
                'outbreak_probability': 0.2
            },
            'pneumonia': {
                'symptoms': ['chest_pain', 'shortness_of_breath', 'cough_with_phlegm', 'fever', 'chills'],
                'severity_levels': ['moderate', 'severe', 'critical'],
                'seasonal_pattern': 'winter',
                'contagious': False,
                'incubation_days': (1, 7),
                'duration_days': (7, 21),
                'outbreak_probability': 0.1
            },
            'food_poisoning': {
                'symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_cramps', 'fever'],
                'severity_levels': ['mild', 'moderate', 'severe'],
                'seasonal_pattern': 'summer',
                'contagious': False,
                'incubation_days': (1, 6),
                'duration_days': (1, 5),
                'outbreak_probability': 0.15
            },
            'meningitis': {
                'symptoms': ['severe_headache', 'neck_stiffness', 'fever', 'confusion', 'sensitivity_to_light'],
                'severity_levels': ['severe', 'critical'],
                'seasonal_pattern': None,
                'contagious': True,
                'incubation_days': (2, 10),
                'duration_days': (7, 14),
                'outbreak_probability': 0.05
            },
            'malaria': {
                'symptoms': ['fever', 'chills', 'sweats', 'headache', 'nausea', 'body_aches'],
                'severity_levels': ['mild', 'moderate', 'severe'],
                'seasonal_pattern': 'rainy',
                'contagious': False,
                'incubation_days': (7, 30),
                'duration_days': (7, 21),
                'outbreak_probability': 0.25
            }
        }
        
        # Geographical regions with population and risk factors
        self.regions = {
            'urban_center': {
                'population': 500000,
                'density': 'high',
                'risk_factors': ['air_pollution', 'crowding'],
                'healthcare_capacity': 'high'
            },
            'suburban_area': {
                'population': 150000,
                'density': 'medium',
                'risk_factors': ['moderate_pollution'],
                'healthcare_capacity': 'medium'
            },
            'rural_district': {
                'population': 50000,
                'density': 'low',
                'risk_factors': ['limited_healthcare'],
                'healthcare_capacity': 'low'
            },
            'industrial_zone': {
                'population': 80000,
                'density': 'medium',
                'risk_factors': ['air_pollution', 'chemical_exposure'],
                'healthcare_capacity': 'medium'
            },
            'coastal_region': {
                'population': 120000,
                'density': 'medium',
                'risk_factors': ['humidity', 'vector_breeding'],
                'healthcare_capacity': 'medium'
            }
        }
        
        # Age groups with different risk profiles
        self.age_groups = {
            'infant': {'min_age': 0, 'max_age': 2, 'risk_multiplier': 1.5},
            'child': {'min_age': 3, 'max_age': 12, 'risk_multiplier': 1.2},
            'teenager': {'min_age': 13, 'max_age': 19, 'risk_multiplier': 0.8},
            'young_adult': {'min_age': 20, 'max_age': 35, 'risk_multiplier': 0.9},
            'middle_aged': {'min_age': 36, 'max_age': 55, 'risk_multiplier': 1.0},
            'senior': {'min_age': 56, 'max_age': 75, 'risk_multiplier': 1.3},
            'elderly': {'min_age': 76, 'max_age': 95, 'risk_multiplier': 1.8}
        }
    
    def generate_patient_demographics(self) -> Dict:
        """Generate realistic patient demographics"""
        age_group = random.choice(list(self.age_groups.keys()))
        age_info = self.age_groups[age_group]
        
        return {
            'patient_id': str(uuid.uuid4()),
            'age': random.randint(age_info['min_age'], age_info['max_age']),
            'age_group': age_group,
            'gender': random.choice(['male', 'female', 'other']),
            'region': random.choice(list(self.regions.keys())),
            'risk_multiplier': age_info['risk_multiplier']
        }
    
    def generate_symptom_description(self, disease: str, severity: str) -> str:
        """Generate natural language symptom descriptions"""
        disease_info = self.diseases[disease]
        symptoms = disease_info['symptoms']
        
        # Select 2-5 symptoms based on severity
        if severity == 'mild':
            num_symptoms = random.randint(2, 3)
        elif severity == 'moderate':
            num_symptoms = random.randint(3, 4)
        else:  # severe/critical
            num_symptoms = random.randint(4, len(symptoms))
        
        selected_symptoms = random.sample(symptoms, min(num_symptoms, len(symptoms)))
        
        # Create natural language variations
        symptom_variations = {
            'fever': ['high temperature', 'fever', 'elevated body temperature', 'feeling hot'],
            'cough': ['persistent cough', 'dry cough', 'coughing', 'hacking cough'],
            'headache': ['severe headache', 'head pain', 'migraine', 'headache'],
            'body_aches': ['muscle pain', 'body aches', 'joint pain', 'muscle soreness'],
            'fatigue': ['extreme tiredness', 'fatigue', 'exhaustion', 'feeling weak'],
            'nausea': ['feeling sick', 'nausea', 'queasiness', 'stomach upset'],
            'vomiting': ['throwing up', 'vomiting', 'being sick', 'retching'],
            'diarrhea': ['loose stools', 'diarrhea', 'frequent bowel movements', 'watery stools'],
            'shortness_of_breath': ['difficulty breathing', 'shortness of breath', 'breathlessness', 'respiratory distress'],
            'chest_pain': ['chest pain', 'chest discomfort', 'pain in chest', 'chest tightness']
        }
        
        # Convert symptoms to natural language
        natural_symptoms = []
        for symptom in selected_symptoms:
            if symptom in symptom_variations:
                natural_symptoms.append(random.choice(symptom_variations[symptom]))
            else:
                natural_symptoms.append(symptom.replace('_', ' '))
        
        # Create sentence patterns
        patterns = [
            "Patient presents with {}",
            "Symptoms include {}",
            "Patient reports {}",
            "Clinical presentation: {}",
            "Patient experiencing {}"
        ]
        
        symptom_text = ', '.join(natural_symptoms[:-1]) + f" and {natural_symptoms[-1]}" if len(natural_symptoms) > 1 else natural_symptoms[0]
        
        return random.choice(patterns).format(symptom_text)
    
    def simulate_outbreak_patterns(self, base_date: datetime, days: int) -> List[Tuple[datetime, str, int]]:
        """Simulate realistic outbreak patterns with clustering"""
        outbreaks = []
        current_date = base_date
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Check for seasonal effects
            month = current_date.month
            is_winter = month in [12, 1, 2]
            is_summer = month in [6, 7, 8]
            is_rainy = month in [6, 7, 8, 9]  # Assuming monsoon season
            
            for disease, info in self.diseases.items():
                base_probability = info['outbreak_probability'] / 365  # Daily probability
                
                # Adjust for seasonal patterns
                if info['seasonal_pattern'] == 'winter' and is_winter:
                    base_probability *= 3
                elif info['seasonal_pattern'] == 'summer' and is_summer:
                    base_probability *= 2.5
                elif info['seasonal_pattern'] == 'rainy' and is_rainy:
                    base_probability *= 2
                
                # Simulate outbreaks
                if random.random() < base_probability:
                    # Determine outbreak size based on disease contagiousness
                    if info['contagious']:
                        outbreak_size = np.random.poisson(10) + 1
                    else:
                        outbreak_size = np.random.poisson(3) + 1
                    
                    outbreaks.append((current_date, disease, outbreak_size))
        
        return outbreaks
    
    def generate_health_reports(self, num_reports: int = 1000, days_range: int = 365) -> pd.DataFrame:
        """Generate comprehensive health reports dataset"""
        reports = []
        base_date = datetime.now() - timedelta(days=days_range)
        
        # Generate outbreak patterns
        outbreaks = self.simulate_outbreak_patterns(base_date, days_range)
        outbreak_dict = {}
        for date, disease, size in outbreaks:
            date_key = date.strftime('%Y-%m-%d')
            if date_key not in outbreak_dict:
                outbreak_dict[date_key] = {}
            outbreak_dict[date_key][disease] = size
        
        for i in range(num_reports):
            # Generate patient demographics
            patient = self.generate_patient_demographics()
            
            # Generate random report date
            report_date = base_date + timedelta(days=random.randint(0, days_range-1))
            date_key = report_date.strftime('%Y-%m-%d')
            
            # Determine disease (with outbreak influence)
            disease_weights = {}
            for disease in self.diseases.keys():
                base_weight = 1.0
                
                # Increase weight if there's an outbreak
                if date_key in outbreak_dict and disease in outbreak_dict[date_key]:
                    base_weight *= (1 + outbreak_dict[date_key][disease] * 0.1)
                
                # Adjust for region and demographics
                region_info = self.regions[patient['region']]
                base_weight *= patient['risk_multiplier']
                
                disease_weights[disease] = base_weight
            
            # Weighted random selection of disease
            diseases = list(disease_weights.keys())
            weights = list(disease_weights.values())
            selected_disease = random.choices(diseases, weights=weights)[0]
            
            # Generate disease details
            disease_info = self.diseases[selected_disease]
            
            # ——— FIXED PART: pick severity with matching weights ———
            levels = disease_info['severity_levels']
            n_levels = len(levels)
            if n_levels == 2:
                # e.g. ['severe', 'critical'] → weight severe:critical = 2:1
                severity_weights = [2, 1]
            elif n_levels == 3:
                severity_weights = [3, 2, 1]
            elif n_levels == 4:
                severity_weights = [4, 3, 2, 1]
            else:
                # fallback: equal weights for any other length
                severity_weights = [1] * n_levels
            
            severity = random.choices(levels, weights=severity_weights)[0]
            # ————————————————————————————————————————————————
            
            # Generate symptom description
            symptom_description = self.generate_symptom_description(selected_disease, severity)
            
            # Generate coordinates (simplified - within region bounds)
            lat_base, lon_base = 40.7128, -74.0060  # Example: NYC coordinates
            lat = lat_base + random.uniform(-0.5, 0.5)
            lon = lon_base + random.uniform(-0.5, 0.5)
            
            report = {
                'report_id': str(uuid.uuid4()),
                'timestamp': report_date,
                'patient_id': patient['patient_id'],
                'age': patient['age'],
                'age_group': patient['age_group'],
                'gender': patient['gender'],
                'region': patient['region'],
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'symptoms_text': symptom_description,
                'actual_disease': selected_disease,  # Ground truth for training
                'severity': severity,
                'is_outbreak_related': date_key in outbreak_dict and selected_disease in outbreak_dict[date_key],
                'healthcare_facility': fake.company() + " Medical Center",
                'reported_by': fake.name(),
                'contact_tracing_needed': disease_info['contagious'] and severity in ['moderate', 'severe', 'critical'],
                'hospitalization_required': severity in ['severe', 'critical'],
                'date': report_date.strftime('%Y-%m-%d'),
                'month': report_date.month,
                'day_of_week': report_date.weekday(),
                'hour': random.randint(6, 22),  # Reports typically during day hours
            }
            
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def generate_intervention_data(self, num_interventions: int = 100) -> pd.DataFrame:
        """Generate synthetic intervention data for AI agent simulation"""
        interventions = []
        
        intervention_types = [
            'quarantine', 'vaccination_campaign', 'public_awareness', 
            'contact_tracing', 'travel_restriction', 'school_closure',
            'mask_mandate', 'social_distancing', 'sanitization_drive'
        ]
        
        for i in range(num_interventions):
            intervention = {
                'intervention_id': str(uuid.uuid4()),
                'type': random.choice(intervention_types),
                'region': random.choice(list(self.regions.keys())),
                'target_disease': random.choice(list(self.diseases.keys())),
                'start_date': fake.date_between(start_date='-1y', end_date='today'),
                'duration_days': random.randint(7, 90),
                'effectiveness_score': random.uniform(0.1, 0.9),
                'cost': random.randint(10000, 1000000),
                'population_affected': random.randint(1000, 100000),
                'compliance_rate': random.uniform(0.3, 0.95)
            }
            interventions.append(intervention)
        
        return pd.DataFrame(interventions)
    
    def save_datasets(self, output_dir: str = "./synthetic_health_data"):
        """Generate and save all datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating health reports...")
        health_reports = self.generate_health_reports(num_reports=5000, days_range=365)
        health_reports.to_csv(f"{output_dir}/health_reports.csv", index=False)
        
        print("Generating intervention data...")
        interventions = self.generate_intervention_data(num_interventions=200)
        interventions.to_csv(f"{output_dir}/interventions.csv", index=False)
        
        # Save metadata
        metadata = {
            'diseases': self.diseases,
            'regions': self.regions,
            'age_groups': self.age_groups,
            'generation_date': datetime.now().isoformat(),
            'total_reports': len(health_reports),
            'date_range': f"{health_reports['timestamp'].min()} to {health_reports['timestamp'].max()}"
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Datasets saved to {output_dir}/")
        print(f"Health reports: {len(health_reports)} records")
        print(f"Interventions: {len(interventions)} records")
        
        return health_reports, interventions

# Function specifically for your LoRA training
def load_your_synthetic_data() -> pd.DataFrame:
    """
    Load synthetic data for ClinicalBERT training
    Returns DataFrame with 'text' and 'label' columns
    """
    generator = SyntheticHealthDataGenerator()
    
    # Generate smaller dataset for initial training
    health_reports = generator.generate_health_reports(num_reports=5000, days_range=365)
    
    # Prepare data for classification training
    training_data = pd.DataFrame({
        'text': health_reports['symptoms_text'],
        'label': health_reports['actual_disease']
    })
    
    return training_data

# Usage example
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticHealthDataGenerator(seed=42)
    
    # Generate and save datasets
    health_reports, interventions = generator.save_datasets()
    
    # Show some statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"Disease distribution:")
    print(health_reports['actual_disease'].value_counts())
    
    print(f"\nSeverity distribution:")
    print(health_reports['severity'].value_counts())
    
    print(f"\nRegional distribution:")
    print(health_reports['region'].value_counts())
    
    print(f"\nOutbreak-related cases: {health_reports['is_outbreak_related'].sum()}")
    
    # Sample records
    print(f"\n=== SAMPLE RECORDS ===")
    print(health_reports[['symptoms_text', 'actual_disease', 'severity', 'region']].head())