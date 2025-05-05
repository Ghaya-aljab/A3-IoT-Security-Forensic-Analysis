#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# In[3]:


#Loading dataset
print("Loading IoT_Weather.csv.xlsx...")
df_weather = pd.read_excel("IoT_Weather.csv.xlsx")

#Adding device type and prediction columns
df_weather['device_type'] = 'Weather'
df_weather['prediction'] = df_weather['type']

#Before preprocessing
print("\n--- Before Preprocessing ---")
print(df_weather.head())
print("Missing values per column:\n", df_weather.isna().sum())

#Drop 'date', 'time' columns if exist
df_weather = df_weather.drop(columns=['date', 'time'], errors='ignore')

#Drop missing values
df_weather = df_weather.dropna()

#Confirm shape after cleaning
print("\nAfter dropping missing values:")
print("Shape:", df_weather.shape)

#Feature Selection
features = ['temperature', 'pressure', 'humidity']
X = df_weather[features]
y_binary = df_weather['label']
y_multiclass = df_weather['prediction']

#Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- After Preprocessing ---")
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print(X_scaled_df.head())

#Split dataset
#$ 
X_train_bin, X_temp_bin, y_train_bin, y_temp_bin = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

X_test_bin, X_val_bin, y_test_bin, y_val_bin = train_test_split(
    X_temp_bin, y_temp_bin, test_size=0.33, random_state=42, stratify=y_temp_bin)

#Multi-class 
X_train_multi, X_temp_multi, y_train_multi, y_temp_multi = train_test_split(
    X_scaled, y_multiclass, test_size=0.3, random_state=42, stratify=y_multiclass)

X_test_multi, X_val_multi, y_test_multi, y_val_multi = train_test_split(
    X_temp_multi, y_temp_multi, test_size=0.33, random_state=42, stratify=y_temp_multi)

print(f"\nBinary - Train: {X_train_bin.shape}, Test: {X_test_bin.shape}, Validation: {X_val_bin.shape}")
print(f"Multi-class - Train: {X_train_multi.shape}, Test: {X_test_multi.shape}, Validation: {X_val_multi.shape}")


# In[ ]:


#part 2:  AI model development and #Part 3: Model evaluation


# In[4]:


print("\n===== Binary Classification (Weather Device) =====")

#RF
rf_bin = RandomForestClassifier(random_state=42)
rf_bin.fit(X_train_bin, y_train_bin)
y_pred_rf_bin = rf_bin.predict(X_test_bin)

#XGBoost
xgb_bin = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_bin.fit(X_train_bin, y_train_bin)
y_pred_xgb_bin = xgb_bin.predict(X_test_bin)

#RF Metrics
acc_rf = accuracy_score(y_test_bin, y_pred_rf_bin)
prec_rf = precision_score(y_test_bin, y_pred_rf_bin)
recall_rf = recall_score(y_test_bin, y_pred_rf_bin)
f1_rf = f1_score(y_test_bin, y_pred_rf_bin)

print("\n--- Random Forest (Binary) ---")
print(classification_report(y_test_bin, y_pred_rf_bin))
sns.heatmap(confusion_matrix(y_test_bin, y_pred_rf_bin), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Binary - Confusion Matrix")
plt.show()

#XGBoost Metrics
acc_xgb = accuracy_score(y_test_bin, y_pred_xgb_bin)
prec_xgb = precision_score(y_test_bin, y_pred_xgb_bin)
recall_xgb = recall_score(y_test_bin, y_pred_xgb_bin)
f1_xgb = f1_score(y_test_bin, y_pred_xgb_bin)

print("\n--- XGBoost (Binary) ---")
print(classification_report(y_test_bin, y_pred_xgb_bin))
sns.heatmap(confusion_matrix(y_test_bin, y_pred_xgb_bin), annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Binary - Confusion Matrix")
plt.show()

#Metric comparison bar graph for Binary
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_scores = [acc_rf, prec_rf, recall_rf, f1_rf]
xgb_scores = [acc_xgb, prec_xgb, recall_xgb, f1_xgb]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest')
rects2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost')

ax.set_ylabel('Score')
ax.set_title('Binary Classification Metrics (Weather Device)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[ ]:


#Multi-class Classification


# In[5]:


print("\n===== Multi-class Classification (Weather Device) =====")

#Encode labels for XGBoost
y_train_multi_encoded = y_train_multi.astype('category').cat.codes
y_test_multi_encoded = y_test_multi.astype('category').cat.codes

#RF
rf_multi = RandomForestClassifier(random_state=42)
rf_multi.fit(X_train_multi, y_train_multi)
y_pred_rf_multi = rf_multi.predict(X_test_multi)

#XGBoost
xgb_multi = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_multi.fit(X_train_multi, y_train_multi_encoded)
y_pred_xgb_multi = xgb_multi.predict(X_test_multi)

#RF Metrics
print("\n--- Random Forest (Multi-class) ---")
print("Accuracy:", accuracy_score(y_test_multi, y_pred_rf_multi))
print(classification_report(y_test_multi, y_pred_rf_multi))
sns.heatmap(confusion_matrix(y_test_multi, y_pred_rf_multi), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Multi-Class - Confusion Matrix")
plt.show()

#XGBoost Metrics
print("\n--- XGBoost (Multi-class) ---")
print("Accuracy:", accuracy_score(y_test_multi_encoded, y_pred_xgb_multi))
print(classification_report(y_test_multi_encoded, y_pred_xgb_multi))
sns.heatmap(confusion_matrix(y_test_multi_encoded, y_pred_xgb_multi), annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Multi-Class - Confusion Matrix")
plt.show()


#Bar Graph 
precisions_rf, recalls_rf, f1s_rf, _ = precision_recall_fscore_support(y_test_multi, y_pred_rf_multi, zero_division=0)

attack_labels = sorted(y_test_multi.unique())

x = np.arange(len(attack_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precisions_rf, width, label='Precision')
rects2 = ax.bar(x, recalls_rf, width, label='Recall')
rects3 = ax.bar(x + width, f1s_rf, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Attack-wise Metrics (Random Forest - Multi-class)')
ax.set_xticks(x)
ax.set_xticklabels(attack_labels, rotation=45)
ax.legend()
fig.tight_layout()
plt.show()

#Bar Graph
precisions_xgb, recalls_xgb, f1s_xgb, _ = precision_recall_fscore_support(y_test_multi_encoded, y_pred_xgb_multi, zero_division=0)

x = np.arange(len(attack_labels))

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precisions_xgb, width, label='Precision')
rects2 = ax.bar(x, recalls_xgb, width, label='Recall')
rects3 = ax.bar(x + width, f1s_xgb, width, label='F1-score')

ax.set_ylabel('Score')
ax.set_title('Attack-wise Metrics (XGBoost - Multi-class)')
ax.set_xticks(x)
ax.set_xticklabels(attack_labels, rotation=45)
ax.legend()
fig.tight_layout()
plt.show()


# In[ ]:


#part 4: validation: 


# In[14]:


#Testing Binary Models on Unseen Data (Validation Set)

print("\n===== Testing on unseen data: Binary Classification =====")

#RF on Validation
y_val_pred_rf_bin = rf_bin.predict(X_val_bin)
print("\n--- Random Forest (Binary, Validation Set) ---")
print("Accuracy:", accuracy_score(y_val_bin, y_val_pred_rf_bin))
print("Classification Report:\n", classification_report(y_val_bin, y_val_pred_rf_bin))
print("Confusion Matrix:\n", confusion_matrix(y_val_bin, y_val_pred_rf_bin))

#XGBoost on Validation
y_val_pred_xgb_bin = xgb_bin.predict(X_val_bin)
print("\n--- XGBoost (Binary, Validation Set) ---")
print("Accuracy:", accuracy_score(y_val_bin, y_val_pred_xgb_bin))
print("Classification Report:\n", classification_report(y_val_bin, y_val_pred_xgb_bin))
print("Confusion Matrix:\n", confusion_matrix(y_val_bin, y_val_pred_xgb_bin))


# In[15]:


#Testing Multi-class Models on Unseen Data (Validation Set) 

print("\n===== Testing on unseen data: Multi-Class Classification =====")

#RF on Validation
y_val_pred_rf_multi = rf_multi.predict(X_val_multi)
print("\n--- Random Forest (Multi-class, Validation Set) ---")
print("Accuracy:", accuracy_score(y_val_multi, y_val_pred_rf_multi))
print("Classification Report:\n", classification_report(y_val_multi, y_val_pred_rf_multi))
print("Confusion Matrix:\n", confusion_matrix(y_val_multi, y_val_pred_rf_multi))

#XGBoost on Validation
y_val_multi_encoded = y_val_multi.astype('category').cat.codes

y_val_pred_xgb_multi = xgb_multi.predict(X_val_multi)
print(accuracy_score(y_val_multi_encoded, y_val_pred_xgb_multi))
print(classification_report(y_val_multi_encoded, y_val_pred_xgb_multi))
print(confusion_matrix(y_val_multi_encoded, y_val_pred_xgb_multi))

