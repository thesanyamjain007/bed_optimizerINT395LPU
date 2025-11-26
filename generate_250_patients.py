import pandas as pd
import numpy as np

df = pd.read_csv("hospital_bed_dataset_250.csv")
print(df["long_stay_label"].value_counts())

np.random.seed(2025)
n = 250

# Patient Info
patient_id = [f"P{str(i+1).zfill(3)}" for i in range(n)]
age = np.random.randint(18, 90, size=n)
gender = np.random.choice(["Male", "Female"], size=n)

# Admission Details
admission_type = np.random.choice(["Emergency", "Planned", "Transfer"], size=n)
department = np.random.choice(
    ["Cardiology", "Orthopedics", "Neurology", "Medicine", "Surgery"],
    size=n
)
diagnosis_category = np.random.choice(
    ["Cardiovascular", "Respiratory", "Infection", "Trauma", "Other"],
    size=n
)
admission_hour = np.random.randint(0, 24, size=n)
day_of_week = np.random.choice(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    size=n
)

# Medical Info
severity_score = np.random.randint(1, 11, size=n)
comorbidities_count = np.random.randint(0, 6, size=n)
previous_admissions = np.random.randint(0, 5, size=n)
icu_flag = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
emergency_flag = (admission_type == "Emergency").astype(int)
surgical_flag = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

# Bed Info
bed_type = np.where(icu_flag == 1, "ICU", np.random.choice(["General", "Private"], size=n))

# Stay logic
base_stay = np.random.randint(1, 5, size=n)
length_of_stay = base_stay.copy()
length_of_stay += (severity_score > 7) * np.random.randint(2, 6, size=n)
length_of_stay += (icu_flag == 1) * np.random.randint(1, 4, size=n)
length_of_stay += (comorbidities_count >= 3) * np.random.randint(1, 3, size=n)

insurance_type = np.random.choice(["Private", "Public", "None"], size=n)

bed_occupancy_level = np.where(
    length_of_stay <= 3,
    "Low",
    np.where(length_of_stay <= 7, "Medium", "High")
)

discharge_within_24h = (length_of_stay <= 1).astype(int)
estimated_stay_days = length_of_stay + np.random.normal(0, 1, size=n).round().astype(int)
estimated_stay_days = np.clip(estimated_stay_days, 1, None)

# ---- IMPORTANT: Add long_stay_label (target for app) ----
# For example: Long stay if > 5 days, otherwise short stay
long_stay_label = (length_of_stay > 5).astype(int)

# Assemble DataFrame
df = pd.DataFrame({
    "patient_id": patient_id,
    "age": age,
    "gender": gender,
    "admission_type": admission_type,
    "department": department,
    "diagnosis_category": diagnosis_category,
    "admission_hour": admission_hour,
    "day_of_week": day_of_week,
    "severity_score": severity_score,
    "comorbidities_count": comorbidities_count,
    "previous_admissions": previous_admissions,
    "icu_flag": icu_flag,
    "emergency_flag": emergency_flag,
    "surgical_flag": surgical_flag,
    "bed_type": bed_type,
    "length_of_stay": length_of_stay,
    "insurance_type": insurance_type,
    "long_stay_label": long_stay_label,       # <-- required for your app!
    "bed_occupancy_level": bed_occupancy_level,
    "discharge_within_24h": discharge_within_24h,
    "estimated_stay_days": estimated_stay_days
})

df.to_csv("hospital_bed_dataset_250.csv", index=False)
print("CSV Saved: hospital_bed_dataset_250.csv, Shape:", df.shape)
print(df.head())
