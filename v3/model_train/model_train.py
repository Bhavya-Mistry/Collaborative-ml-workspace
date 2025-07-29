import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
import joblib

# Read data
df = pd.read_csv(r"data/19k.csv")

# Define columns for multi-label encoding
multi_cols = [
    'Programming_Languages',
    'Certifications',
    'Extracurricular_Interests',
    'Interest_Areas',
    'Soft_Skills',
    'Tools_Techstack',
    'Favourite_Subjects',
    'Problem_Solving_Style'
]

# Fill NaN values in multi-label columns
for col in multi_cols:
    df[col] = df[col].fillna('')

# Split the multi-label columns into lists
def split_to_list(s):
    return [item.strip() for item in s.split(',') if item.strip()]

for col in multi_cols:
    df[col] = df[col].apply(split_to_list)

# Apply MultiLabelBinarizer and OneHotEncoder to the dataframe
mlb_dict = {}
for col in multi_cols:
    mlb = MultiLabelBinarizer()
    mat = mlb.fit_transform(df[col])
    cols = [f"{col}_{v}" for v in mlb.classes_]
    df_mlb = pd.DataFrame(mat, columns=cols, index=df.index)
    df = pd.concat([df, df_mlb], axis=1).drop(columns=[col])
    mlb_dict[col] = mlb  # Save the mlb for future use

# One-hot encoding for the 'Preferred_Work_Style'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
arr = ohe.fit_transform(df[['Preferred_Work_Style']])
ohe_cols = ohe.get_feature_names_out(['Preferred_Work_Style'])
df_ohe = pd.DataFrame(arr, columns=ohe_cols, index=df.index)
df = pd.concat([df, df_ohe], axis=1).drop(columns=['Preferred_Work_Style'])

# Prepare features (X) and target (y)
X = df.drop(columns=['Recommended_Career'])
y = df['Recommended_Career']

# Label encode the target variable
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train RandomForestClassifier
final_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,            # cap tree depth
    min_samples_leaf=2,      # force at least 2 samples per leaf
    ccp_alpha=1e-3,          # cost‑complexity pruning
    random_state=42
)

final_rf.fit(X, y_enc)

# Save model, encoders, and transformers
joblib.dump(final_rf, r"trained-models\careermodel.pkl")
joblib.dump(le, r"trained-models\labelencoder.pkl")
joblib.dump(mlb_dict, r"trained-models\mlbdict.pkl")
joblib.dump(ohe, r"trained-models\ohencoder.pkl")

# To confirm the process, we can also plot feature importances (optional)
import matplotlib.pyplot as plt

# Get importances & sort
importances = pd.Series(final_rf.feature_importances_, index=X.columns)
top20 = importances.nlargest(20)

plt.figure(figsize=(10, 6))
top20.sort_values().plot.barh()
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.show()
