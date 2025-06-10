import os
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker for generating realistic data
fake = Faker()

class SyntheticHealthDataGenerator:
    """
    Synthetic health plus intervention data generator with realistic dates (including up to 2025).
    Covid-19 outbreaks taper off post-2023; other diseases follow seasonal patterns.
    Generates and merges health reports and intervention data into one dataset.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

        # Disease definitions with seasonal patterns and outbreak probabilities
        self.diseases = {
            'influenza': {
                'symptoms': ['fever','cough','headache','body_aches','fatigue','chills'],
                'severity_levels': ['mild','moderate','severe'],
                'seasonal_pattern': 'winter',
                'contagious': True,
                'base_probability': 0.3
            },
            'covid19': {
                'symptoms': ['fever','dry_cough','shortness_of_breath','loss_of_taste','loss_of_smell','fatigue'],
                'severity_levels': ['mild','moderate','severe','critical'],
                'seasonal_pattern': None,
                'contagious': True,
                'base_probability': 0.4,
                'post_2023_factor': 0.2
            },
            'gastroenteritis': {
                'symptoms': ['nausea','vomiting','diarrhea','abdominal_pain','fever'],
                'severity_levels': ['mild','moderate','severe'],
                'seasonal_pattern': 'summer',
                'contagious': True,
                'base_probability': 0.2
            },
            'pneumonia': {
                'symptoms': ['chest_pain','shortness_of_breath','cough_with_phlegm','fever','chills'],
                'severity_levels': ['moderate','severe','critical'],
                'seasonal_pattern': 'winter',
                'contagious': False,
                'base_probability': 0.1
            },
            'food_poisoning': {
                'symptoms': ['nausea','vomiting','diarrhea','abdominal_cramps','fever'],
                'severity_levels': ['mild','moderate','severe'],
                'seasonal_pattern': 'summer',
                'contagious': False,
                'base_probability': 0.15
            },
            'meningitis': {
                'symptoms': ['severe_headache','neck_stiffness','fever','confusion','sensitivity_to_light'],
                'severity_levels': ['severe','critical'],
                'seasonal_pattern': None,
                'contagious': True,
                'base_probability': 0.05
            },
            'malaria': {
                'symptoms': ['fever','chills','sweats','headache','nausea','body_aches'],
                'severity_levels': ['mild','moderate','severe'],
                'seasonal_pattern': 'rainy',
                'contagious': False,
                'base_probability': 0.25
            }
        }

        # Geographical regions with population and risk factors
        self.regions = {
            'urban_center':    {'population': 500_000, 'density': 'high',   'risk_factors': ['air_pollution','crowding'],           'healthcare_capacity': 'high'},
            'suburban_area':   {'population': 150_000, 'density': 'medium', 'risk_factors': ['moderate_pollution'],                'healthcare_capacity': 'medium'},
            'rural_district':  {'population':  50_000, 'density': 'low',    'risk_factors': ['limited_healthcare'],               'healthcare_capacity': 'low'},
            'industrial_zone': {'population':  80_000, 'density': 'medium', 'risk_factors': ['air_pollution','chemical_exposure'], 'healthcare_capacity': 'medium'},
            'coastal_region':  {'population': 120_000, 'density': 'medium', 'risk_factors': ['humidity','vector_breeding'],      'healthcare_capacity': 'medium'}
        }

        # Age groups with risk multipliers
        self.age_groups = {
            'infant':      {'min_age':0,'max_age':2, 'risk_multiplier':1.5},
            'child':       {'min_age':3,'max_age':12,'risk_multiplier':1.2},
            'teenager':    {'min_age':13,'max_age':19,'risk_multiplier':0.8},
            'young_adult': {'min_age':20,'max_age':35,'risk_multiplier':0.9},
            'middle_aged': {'min_age':36,'max_age':55,'risk_multiplier':1.0},
            'senior':      {'min_age':56,'max_age':75,'risk_multiplier':1.3},
            'elderly':     {'min_age':76,'max_age':95,'risk_multiplier':1.8}
        }

    def generate_patient_demographics(self) -> Dict:
        age_group = random.choice(list(self.age_groups.keys()))
        info = self.age_groups[age_group]
        return {
            'patient_id': str(uuid.uuid4()),
            'age': random.randint(info['min_age'], info['max_age']),
            'age_group': age_group,
            'gender': random.choice(['male','female','other']),
            'region': random.choice(list(self.regions.keys())),
            'risk_multiplier': info['risk_multiplier']
        }

    def generate_symptom_description(self, disease: str, severity: str) -> str:
        info = self.diseases[disease]
        syms = info['symptoms']
        if severity == 'mild': num = random.randint(2,3)
        elif severity == 'moderate': num = random.randint(3,4)
        else: num = random.randint(4, len(syms))
        selected = random.sample(syms, num)
        variations = {
            'fever':['high temperature','fever','elevated temperature','feeling hot'],
            'cough':['persistent cough','dry cough','coughing'],
            'headache':['severe headache','migraine','head pain'],
            'body_aches':['body aches','muscle pain'],
            'fatigue':['fatigue','exhaustion'],
            'nausea':['nausea','queasiness'],
            'vomiting':['vomiting','being sick'],
            'diarrhea':['diarrhea','watery stools'],
            'shortness_of_breath':['shortness of breath','breathlessness'],
            'chest_pain':['chest pain','chest discomfort']
        }
        nat = [random.choice(variations.get(s, [s.replace('_',' ')])) for s in selected]
        text = ', '.join(nat[:-1]) + ' and ' + nat[-1] if len(nat)>1 else nat[0]
        pattern = random.choice(['Patient presents with {}','Symptoms include {}','Clinical presentation: {}'])
        return pattern.format(text)

    def simulate_outbreak_patterns(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, str, int]]:
        out = []
        cur = start_date
        while cur <= end_date:
            m, y = cur.month, cur.year
            winter = m in [12,1,2]
            summer = m in [6,7,8]
            rainy  = m in [6,7,8,9]
            for d, info in self.diseases.items():
                prob = info['base_probability'] / 365
                if info['seasonal_pattern']=='winter' and winter: prob *= 3
                if info['seasonal_pattern']=='summer' and summer: prob *= 2.5
                if info['seasonal_pattern']=='rainy' and rainy: prob *= 2
                if d=='covid19' and y>=2024: prob *= info.get('post_2023_factor',1)
                if random.random() < prob:
                    size = (np.random.poisson(10)+1) if info['contagious'] else (np.random.poisson(3)+1)
                    out.append((cur, d, size))
            cur += timedelta(days=1)
        return out

    def generate_health_reports(self, start_date: datetime, end_date: datetime,
                                num_reports: int = 1000, balanced: bool = False,
                                outbreak_fraction: float = 0.5) -> pd.DataFrame:
        days = (end_date - start_date).days
        outbreaks = self.simulate_outbreak_patterns(start_date, end_date)
        outbreak_index: Dict[str, Dict[str, int]] = {}
        for dt, disease, size in outbreaks:
            key = dt.strftime('%Y-%m-%d')
            outbreak_index.setdefault(key, {})[disease] = size

        records = []
        for _ in range(num_reports):
            pat = self.generate_patient_demographics()
            rep_dt = start_date + timedelta(days=random.randint(0, days))
            date_str = rep_dt.strftime('%Y-%m-%d')

            weights: Dict[str, float] = {}
            for disease, info in self.diseases.items():
                base_w = 1 + pat['risk_multiplier']
                multiplier = 1 + (outbreak_index.get(date_str, {}).get(disease, 0) * 0.1)
                weights[disease] = base_w * multiplier

            selected = random.choices(list(weights), weights=list(weights.values()))[0]
            severity_levels = self.diseases[selected]['severity_levels']
            severity = random.choices(severity_levels, weights=range(len(severity_levels), 0, -1))[0]

            records.append({
                'timestamp': rep_dt,
                'age': pat['age'],
                'age_group': pat['age_group'],
                'gender': pat['gender'],
                'region': pat['region'],
                'latitude': round(40.7128 + random.uniform(-0.5, 0.5), 6),
                'longitude': round(-74.0060 + random.uniform(-0.5, 0.5), 6),
                'symptoms_text': self.generate_symptom_description(selected, severity),
                'actual_disease': selected,
                'severity': severity,
                'is_outbreak_related': selected in outbreak_index.get(date_str, {}),
                'contact_tracing_needed': info['contagious'] and severity in ['moderate','severe','critical'],
                'hospitalization_required': severity in ['severe','critical'],
                'date': date_str,
                'month': rep_dt.month,
                'day_of_week': rep_dt.weekday(),
                'hour': random.randint(6,22)
            })
        df = pd.DataFrame(records)
        if balanced:
            df_out = df[df['is_outbreak_related']]
            df_non = df[~df['is_outbreak_related']]
            n_out = int(num_reports * outbreak_fraction)
            n_non = num_reports - n_out
            df_out = df_out.sample(n_out, replace=len(df_out)<n_out, random_state=1)
            df_non = df_non.sample(n_non, replace=len(df_non)<n_non, random_state=1)
            df = pd.concat([df_out, df_non]).sample(frac=1, random_state=1).reset_index(drop=True)
        return df

    def generate_intervention_data(
        self,
        num_interventions: int = 100,
        start_window: Tuple[datetime, datetime] = (datetime(2020,1,1), datetime(2025,6,1))
    ) -> pd.DataFrame:
        """
        Generate interventions with disease-specific realistic start windows.
        uses num_interventions and a default global window for others.
        """
        types = ['quarantine','vaccination_campaign','public_awareness','contact_tracing',
                 'travel_restriction','school_closure','mask_mandate','social_distancing','sanitization_drive']
        rows = []
        for _ in range(num_interventions):
            disease = random.choice(list(self.diseases.keys()))
            # define window start/end based on disease
            if disease == 'covid19':
                ws, we = datetime(2020,1,1), datetime(2022,12,31)
            elif disease == 'influenza':
                year = random.randint(2020,2025)
                month = random.choice([12,1,2]); day=random.randint(1,28)
                ws = datetime(year,month,day); we=ws+timedelta(days=random.randint(7,30))
            elif disease == 'gastroenteritis':
                year=random.randint(2020,2025)
                month=random.choice([6,7,8]); day=random.randint(1,28)
                ws=datetime(year,month,day); we=ws+timedelta(days=random.randint(7,30))
            elif disease == 'malaria':
                year=random.randint(2020,2025)
                month=random.choice([6,7,8,9]); day=random.randint(1,28)
                ws=datetime(year,month,day); we=ws+timedelta(days=random.randint(7,30))
            else:
                ws, we = start_window
            total_days = max((we - ws).days, 0)
            int_start = ws + timedelta(days=random.randint(0, total_days))

            rows.append({
                'intervention_id': str(uuid.uuid4()),
                'type': random.choice(types),
                'region': random.choice(list(self.regions.keys())),
                'target_disease': disease,
                'start_date': int_start,
                'duration_days': random.randint(7,90),
                'effectiveness_score': random.uniform(0.1,0.9),
                'cost': random.randint(10000,1000000),
                'population_affected': random.randint(1000,100000),
                'compliance_rate': random.uniform(0.3,0.95)
            })
        return pd.DataFrame(rows)

    def save_combined_dataset(self, output_path: str = 'dataset/health_reports_data.csv',
                              start_date: datetime = datetime(2020,1,1),
                              end_date: datetime = datetime(2025,6,1),
                              num_reports: int = 10000, num_interventions: int = 200,
                              balanced: bool = True, outbreak_fraction: float = 0.4) -> pd.DataFrame:
        hr = self.generate_health_reports(start_date, end_date, num_reports, balanced, outbreak_fraction)
        iv = self.generate_intervention_data(num_interventions)
        iv_grp = iv.groupby(['region','target_disease'])

        assigned = []
        for idx, row in hr.iterrows():
            key = (row['region'], row['actual_disease'])
            if key in iv_grp.groups:
                choice = iv_grp.get_group(key).sample(1, random_state=idx).iloc[0]
            else:
                choice = iv.sample(1, random_state=idx).iloc[0]
            assigned.append(choice)

        iv_assigned = pd.DataFrame(assigned).add_prefix('intervention_')
        combined = pd.concat([hr.reset_index(drop=True), iv_assigned.reset_index(drop=True)], axis=1)
        combined.to_csv(output_path, index=False)

        print(f"Saved combined dataset to {output_path}, shape={combined.shape}")
        print(combined.head())
        return combined

if __name__=='__main__':
    gen = SyntheticHealthDataGenerator(seed=42)
    gen.save_combined_dataset()