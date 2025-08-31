import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Preprocessing
test_ids = test['id']
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

X = train.drop('y', axis=1)
y = train['y']

# Feature Engineering 
def create_features(df):
    df = df.copy()
    
    #
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'y' in numerical_cols:
        numerical_cols.remove('y')
    
    print("Numerical columns available:", numerical_cols)
    
 
    tenure_cols = [col for col in numerical_cols if 'tenure' in col.lower() or 'month' in col.lower()]
    charge_cols = [col for col in numerical_cols if 'charge' in col.lower() or 'fee' in col.lower()]
    
    if tenure_cols and charge_cols:
        # Create average monthly charge
        df['avg_monthly_charge'] = df[charge_cols[0]] / df[tenure_cols[0]] if tenure_cols[0] != 0 else 0
    
    # Create polynomial features for important numerical columns
    for col in numerical_cols[:3]:  # Just use first 3 numerical columns
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_log'] = np.log1p(df[col])
    
    return df

X = create_features(X)
test = create_features(test)

# Encode categorical variables
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Model parameters 
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
}

# Cross-validation
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    model = LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(500, verbose=False),
            lgb.log_evaluation(False)
        ]
    )
    
    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    sub_preds += model.predict_proba(test)[:, 1] / NFOLDS
    
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f'Fold {n_fold+1} AUC: {fold_auc:.6f}')

print(f'\nOverall CV AUC: {roc_auc_score(y, oof_preds):.6f}')

# Generate submission
submission = pd.DataFrame({'id': test_ids, 'y': sub_preds})
submission.to_csv('submission.csv', index=False)
