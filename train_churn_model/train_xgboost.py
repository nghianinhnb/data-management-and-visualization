import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'xgboost'
N_TRIALS = 50
N_FOLDS = 5
N_JOBS = 4
RANDOM_STATE = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading customer-level dataset...")
try:
    df = pd.read_csv('customer_level_features.csv', parse_dates=['last_purchase_timestamp'])
    print(f"Loaded: {df.shape}")
except FileNotFoundError:
    print("Error: customer_level_features.csv not found")
    print("Run feature_engineering_customer_level.py first")
    exit(1)

df_sorted = df.sort_values('last_purchase_timestamp').reset_index(drop=True)
print(f"Date range: {df_sorted['last_purchase_timestamp'].min()} to {df_sorted['last_purchase_timestamp'].max()}")

print("\nDefining features...")
numerical_features = [
    'avg_delivery_deviation', 'avg_seller_delivery_deviation',
    'avg_freight_value', 'avg_product_name_length', 'avg_delivery_distance',
    'avg_processing_time', 'avg_shipping_time', 'avg_price_per_weight',
    'avg_freight_to_price_ratio', 'avg_price_deviation_from_category',
    'avg_product_complexity_score',
    'recency', 'frequency', 'monetary_total', 'monetary_mean',
    'monetary_std', 'monetary_max', 'monetary_min',
    'std_delivery_deviation', 'total_late_deliveries', 'late_delivery_rate',
    'total_freight_value', 'avg_days_between_orders', 'std_days_between_orders',
    'min_days_between_orders', 'max_days_between_orders',
    'unique_products_purchased', 'unique_categories_purchased',
    'avg_product_weight', 'avg_product_photos',
    'avg_review_score', 'std_review_score', 'min_review_score', 'has_low_review',
    'avg_payment_installments', 'max_payment_installments',
    'avg_payment_value', 'total_payment_value',
    'total_cross_state_deliveries', 'cross_state_delivery_rate',
    'is_customer_from_capital', 'capital_purchase_rate',
    'most_common_purchase_hour', 'most_common_purchase_day',
    'weekend_purchase_rate', 'business_hours_purchase_rate',
    'avg_delivery_distance_per_day', 'unique_sellers'
]

categorical_features = [
    'customer_state', 'customer_city', 'most_frequent_seller_state',
    'most_frequent_order_status', 'most_frequent_payment_type',
    'most_frequent_category'
]

numerical_features = [f for f in numerical_features if f in df_sorted.columns]
categorical_features = [f for f in categorical_features if f in df_sorted.columns]
all_features = numerical_features + categorical_features
target = 'churn'
all_features = [f for f in all_features if f != 'last_purchase_timestamp']

print(f"Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}, Total: {len(all_features)}")

print("\nPerforming stratified three-way split...")
df_model = df_sorted[all_features + [target]].copy()
initial_rows = len(df_model)
df_model = df_model.dropna(subset=[target])
dropped_rows = initial_rows - len(df_model)
if dropped_rows > 0:
    print(f"Dropped {dropped_rows} rows with missing target")

y = df_model[target]
X = df_model.drop(columns=[target])

# Adjust for stratification
train_val_size = 1 - TEST_RATIO
val_size_adjusted = VAL_RATIO / train_val_size

# Split into train/validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
)

# Split train/validation set into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_val, y_train_val, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_train_val
)

# Recreate dataframes
train_df = pd.concat([X_train, y_train], axis=1)
validation_df = pd.concat([X_validation, y_validation], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

n_samples = len(df_model)
print(f"Train: {len(train_df):,} ({len(train_df)/n_samples:.1%}), churn {train_df[target].mean():.2%}")
print(f"Validation: {len(validation_df):,} ({len(validation_df)/n_samples:.1%}), churn {validation_df[target].mean():.2%}")
print(f"Test: {len(test_df):,} ({len(test_df)/n_samples:.1%}), churn {test_df[target].mean():.2%}")

print("\nPreparing data...")

boolean_columns = ['has_low_review', 'is_customer_from_capital']
for col in boolean_columns:
    if col in all_features:
        for df_temp in [train_df, validation_df, test_df]:
            df_temp[col] = df_temp[col].replace({'True': 1, 'False': 0, True: 1, False: 0})
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0).astype(int)

train_medians = {}
for col in numerical_features:
    train_medians[col] = train_df[col].median()
    train_df[col].fillna(train_medians[col], inplace=True)
    validation_df[col].fillna(train_medians[col], inplace=True)
    test_df[col].fillna(train_medians[col], inplace=True)

for col in categorical_features:
    train_df[col].fillna('MISSING', inplace=True)
    validation_df[col].fillna('MISSING', inplace=True)
    test_df[col].fillna('MISSING', inplace=True)

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le
    validation_df[col] = validation_df[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    test_df[col] = test_df[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

for col in all_features:
    if col in numerical_features:
        for df_temp in [train_df, validation_df, test_df]:
            df_temp[col] = df_temp[col].replace([np.inf, -np.inf], np.nan)
            if df_temp[col].isnull().sum() > 0:
                df_temp[col].fillna(train_medians.get(col, df_temp[col].median()), inplace=True)

X_train = train_df[all_features].values
y_train = train_df[target].values
X_validation = validation_df[all_features].values
y_validation = validation_df[target].values
X_test = test_df[all_features].values
y_test = test_df[target].values

print(f"X_train: {X_train.shape}, X_validation: {X_validation.shape}, X_test: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance: {scale_pos_weight:.2f}")

print(f"\nSetting up Optuna optimization with StratifiedKFold ({N_TRIALS} trials)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': N_JOBS,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train, verbose=False)

        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        f1_scores.append(best_f1)

    return np.mean(f1_scores)

print("\nRunning optimization...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_score = study.best_value

print(f"\nOptimization complete")
print(f"Best CV F1: {best_score:.4f}")
print(f"\nBest Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nModel selection on validation set...")
final_params = best_params.copy()
final_params['scale_pos_weight'] = scale_pos_weight
final_params['n_jobs'] = N_JOBS
final_params['random_state'] = RANDOM_STATE
final_params['eval_metric'] = 'logloss'

model_train = XGBClassifier(**final_params)
model_train.fit(X_train, y_train)

y_pred_proba_val = model_train.predict_proba(X_validation)[:, 1]

best_f1_val = 0
best_threshold = 0.5

for threshold in np.arange(0.05, 0.95, 0.01):
    y_pred = (y_pred_proba_val >= threshold).astype(int)
    if y_pred.sum() > 0:
        f1 = f1_score(y_validation, y_pred, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_threshold = threshold

y_pred_val = (y_pred_proba_val >= best_threshold).astype(int)
roc_auc_val = roc_auc_score(y_validation, y_pred_proba_val)

print(f"\nValidation performance:")
print(f"  ROC-AUC: {roc_auc_val:.4f}")
print(f"  F1: {best_f1_val:.4f}")
print(f"  Threshold: {best_threshold:.2f}")

print("\nFinal evaluation on test set...")
X_train_combined = np.vstack([X_train, X_validation])
y_train_combined = np.concatenate([y_train, y_validation])

final_model = XGBClassifier(**final_params)
final_model.fit(X_train_combined, y_train_combined)

y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba_test >= best_threshold).astype(int)

roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
f1_test = f1_score(y_test, y_pred_test)

print(f"\nTest performance:")
print(f"  ROC-AUC: {roc_auc_test:.4f}")
print(f"  F1: {f1_test:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred_test, target_names=['No Churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion matrix:")
print(cm)
print(f"  TN: {cm[0,0]:,}, FP: {cm[0,1]:,}, FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")

print("\nSaving results...")
model_filename = os.path.join(OUTPUT_DIR, 'model.pkl')
joblib.dump(final_model, model_filename)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
joblib.dump(train_medians, os.path.join(OUTPUT_DIR, 'train_medians.pkl'))

best_params_df = pd.DataFrame([best_params])
best_params_df['best_cv_f1'] = best_score
best_params_df['validation_f1'] = best_f1_val
best_params_df['validation_auc'] = roc_auc_val
best_params_df['test_f1'] = f1_test
best_params_df['test_auc'] = roc_auc_test
best_params_df['optimal_threshold'] = best_threshold
best_params_df.to_csv(os.path.join(OUTPUT_DIR, 'best_params.csv'), index=False)

importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

print("\nTop 20 features:")
print(importance_df.head(20).to_string(index=False))

print(f"\nComplete. Test F1: {f1_test:.4f}, Test AUC: {roc_auc_test:.4f}")
