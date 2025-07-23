import pandas as pd

df = pd.read_csv(r"career-navigator\v1\data\dataset.csv")
print(df.columns.tolist())


df.columns
multi_label_cols = [
    'Interest_Areas',
    'Soft_Skills',
    'Programming_Languages',
    'Tools_Techstack',  # <-- ✅ fixed
    'Certifications',   # <-- ✅ match with actual column name
    'Extracurricular_Interests',
    'Favourite_Subjects'
]



from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Utility: split multi-label strings
def split_multi_label(col):
    return col.fillna('').apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])

# 1. Encode all multi-label columns
def encode_multi_label(df, cols):
    mlb_dict = {}
    new_df_parts = []

    for col in cols:
        mlb = MultiLabelBinarizer()
        labels = split_multi_label(df[col])
        transformed = mlb.fit_transform(labels)
        mlb_df = pd.DataFrame(transformed, columns=[f"{col}_{c}" for c in mlb.classes_])
        new_df_parts.append(mlb_df)
        mlb_dict[col] = mlb

    return pd.concat(new_df_parts, axis=1), mlb_dict

# 2. Label encode single-class categorical
def encode_label(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].fillna("Unknown"))
    return le

# 3. Binary mapping
def map_binary(df, col):
    return df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0)

# Preprocess all
def preprocess_dataset(df):
    multi_label_encoded, mlb_dict = encode_multi_label(df, multi_label_cols)

    le_work_style = encode_label(df, 'Preferred_Work_Style')
    le_problem_style = encode_label(df, 'Problem_Solving_Style')

    df['Wants_to_Go_for_Masters'] = map_binary(df, 'Wants_to_Go_for_Masters')
    df['Interested_in_Research'] = map_binary(df, 'Interested_in_Research')

    df['CGPA'] = df['CGPA'].fillna(df['CGPA'].mean())
    df['Current_Projects_Count'] = df['Current_Projects_Count'].fillna(0)
    df['Internship_Experience'] = df['Internship_Experience'].fillna(0)

    X = pd.concat([
        multi_label_encoded,
        df[['Preferred_Work_Style', 'Problem_Solving_Style', 'Wants_to_Go_for_Masters',
            'Interested_in_Research', 'CGPA', 'Current_Projects_Count', 'Internship_Experience']]
    ], axis=1)

    label_encoder = LabelEncoder()
    df['Recommended_Career'] = df['Recommended_Career'].fillna('Unknown')
    label_encoder.fit(df['Recommended_Career'])
    y = label_encoder.transform(df['Recommended_Career'])

    return X, y, mlb_dict, label_encoder



X, y, mlb_dict, label_encoder = preprocess_dataset(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))


import joblib

joblib.dump(model, r"career-navigator\v1\saved-models\career_model.pkl")
joblib.dump(mlb_dict, r"career-navigator\v1\saved-models\mlb_dict.pkl")
joblib.dump(label_encoder, r"career-navigator\v1\saved-models\label_encoder.pkl")
