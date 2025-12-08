import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

print("Checking available model results...")

results_files = {
    'XGBoost': 'xgboost/best_params.csv',
    'CatBoost': 'catboost/best_params.csv',
    'LightGBM': 'lightgbm/best_params.csv'
}

available_models = {}
for model_name, filepath in results_files.items():
    if os.path.exists(filepath):
        available_models[model_name] = filepath
        print(f"  Found {model_name}: {filepath}")
    else:
        print(f"  Missing {model_name}: {filepath}")

if len(available_models) == 0:
    print("\nNo model results found")
    print("Train at least one model first:")
    print("  python train_xgboost_optuna.py")
    print("  python train_catboost.py")
    print("  python train_lightgbm.py")
    exit(1)

print(f"\nFound {len(available_models)} trained models")

print("\nLoading and comparing results...")
comparison_data = []

for model_name, filename in available_models.items():
    df = pd.read_csv(filename)

    if len(df) > 0:
        row = df.iloc[0]

        cv_f1 = row.get('best_cv_f1', row.get('cv_f1', 0))
        validation_f1 = row.get('validation_f1', 0)
        validation_auc = row.get('validation_auc', 0)
        test_f1 = row.get('test_f1', 0)
        test_auc = row.get('test_auc', 0)
        threshold = row.get('optimal_threshold', row.get('threshold', 0.5))

        comparison_data.append({
            'Model': model_name,
            'CV F1': cv_f1,
            'Val F1': validation_f1,
            'Val AUC': validation_auc,
            'Test F1': test_f1,
            'Test AUC': test_auc,
            'Threshold': threshold
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test F1', ascending=False)

print("\nModel performance comparison:")
print(comparison_df.to_string(index=False))

best_model_val_idx = comparison_df['Val F1'].idxmax()
best_model_val = comparison_df.loc[best_model_val_idx, 'Model']
best_val_f1 = comparison_df.loc[best_model_val_idx, 'Val F1']

best_model_test_idx = comparison_df['Test F1'].idxmax()
best_model_test = comparison_df.loc[best_model_test_idx, 'Model']
best_test_f1 = comparison_df.loc[best_model_test_idx, 'Test F1']
best_test_auc = comparison_df.loc[best_model_test_idx, 'Test AUC']

print(f"\nBest model (by validation set): {best_model_val}")
print(f"  Validation F1: {best_val_f1:.4f}")

print(f"\nBest model (by test set): {best_model_test}")
print(f"  Test F1: {best_test_f1:.4f}")
print(f"  Test AUC: {best_test_auc:.4f}")

if best_model_val == best_model_test:
    print(f"\nModel selection consistent: {best_model_val} performs best on both sets")
else:
    print(f"\nModel selection differs: {best_model_val} (validation) vs {best_model_test} (test)")
    print(f"Recommend using: {best_model_val} (selected on validation set)")

print("\nCreating comparison visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax1 = axes[0, 0]
x_pos = range(len(comparison_df))
width = 0.35

bars_val = ax1.bar([i - width/2 for i in x_pos], comparison_df['Val F1'],
                    width, label='Validation F1', color='lightgreen', edgecolor='black')
bars_test = ax1.bar([i + width/2 for i in x_pos], comparison_df['Test F1'],
                     width, label='Test F1', color='lightblue', edgecolor='black')

best_val_pos = comparison_df['Val F1'].idxmax()
bars_val[best_val_pos].set_edgecolor('darkred')
bars_val[best_val_pos].set_linewidth(3)

ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('Validation vs Test F1 Score', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1.0)

ax2 = axes[0, 1]
bars_val_auc = ax2.bar([i - width/2 for i in x_pos], comparison_df['Val AUC'],
                        width, label='Validation AUC', color='lightcoral', edgecolor='black')
bars_test_auc = ax2.bar([i + width/2 for i in x_pos], comparison_df['Test AUC'],
                         width, label='Test AUC', color='lightsalmon', edgecolor='black')

bars_val_auc[best_val_pos].set_edgecolor('darkred')
bars_val_auc[best_val_pos].set_linewidth(3)

ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.set_title('Validation vs Test ROC-AUC', fontsize=14)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.0)

ax3 = axes[1, 0]
width2 = 0.25
bars_cv = ax3.bar([i - width2 for i in x_pos], comparison_df['CV F1'],
                   width2, label='CV F1', color='#90EE90', edgecolor='black')
bars_val = ax3.bar(x_pos, comparison_df['Val F1'],
                    width2, label='Val F1', color='#FFD700', edgecolor='black')
bars_test = ax3.bar([i + width2 for i in x_pos], comparison_df['Test F1'],
                     width2, label='Test F1', color='#87CEEB', edgecolor='black')

ax3.set_ylabel('F1 Score', fontsize=12)
ax3.set_title('CV / Validation / Test F1 Progression', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 1.0)

ax4 = axes[1, 1]
cv_to_val_gap = comparison_df['CV F1'] - comparison_df['Val F1']
val_to_test_gap = comparison_df['Val F1'] - comparison_df['Test F1']

bars_gap1 = ax4.bar([i - width/2 for i in x_pos], cv_to_val_gap,
                     width, label='CV to Val Gap', color='orange', edgecolor='black', alpha=0.7)
bars_gap2 = ax4.bar([i + width/2 for i in x_pos], val_to_test_gap,
                     width, label='Val to Test Gap', color='purple', edgecolor='black', alpha=0.7)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)

ax4.set_ylabel('Performance Gap', fontsize=12)
ax4.set_title('Generalization Gap Analysis', fontsize=14)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comparison plot: model_comparison.png")
plt.close()

print("\nGeneralization analysis:")
for idx, row in comparison_df.iterrows():
    cv_to_val_gap = row['CV F1'] - row['Val F1']
    val_to_test_gap = row['Val F1'] - row['Test F1']
    total_gap = row['CV F1'] - row['Test F1']
    model_name = row['Model']

    print(f"\n{model_name}:")
    print(f"  CV F1: {row['CV F1']:.4f}")
    print(f"  Val F1: {row['Val F1']:.4f} (gap from CV: {cv_to_val_gap:+.4f})")
    print(f"  Test F1: {row['Test F1']:.4f} (gap from Val: {val_to_test_gap:+.4f})")
    print(f"  Total gap: {total_gap:+.4f}")

    if abs(val_to_test_gap) < 0.02:
        status = "Excellent generalization"
    elif abs(val_to_test_gap) < 0.05:
        status = "Good generalization"
    elif val_to_test_gap > 0.05:
        status = "Overfitting detected"
    else:
        status = "Underfitting"

    print(f"  Status: {status}")

comparison_df.to_csv('model_comparison_results.csv', index=False)
print(f"\nComparison results saved: model_comparison_results.csv")

print("\nRecommended model (selected on validation set):")
print(f"  Model: {best_model_val}")
print(f"  Validation F1: {best_val_f1:.4f}")
print(f"  Test F1: {comparison_df.loc[best_model_val_idx, 'Test F1']:.4f}")
print(f"  Test ROC-AUC: {comparison_df.loc[best_model_val_idx, 'Test AUC']:.4f}")
print(f"  Threshold: {comparison_df.loc[best_model_val_idx, 'Threshold']:.2f}")

print(f"\nBest performing model (on test set):")
print(f"  Model: {best_model_test}")
print(f"  Test F1: {best_test_f1:.4f}")
print(f"  Test ROC-AUC: {best_test_auc:.4f}")

print(f"\nPerformance range (test set):")
print(f"  F1: {comparison_df['Test F1'].min():.4f} to {comparison_df['Test F1'].max():.4f}")
print(f"  ROC-AUC: {comparison_df['Test AUC'].min():.4f} to {comparison_df['Test AUC'].max():.4f}")

if comparison_df['Test F1'].min() > 0:
    improvement = (best_test_f1 - comparison_df['Test F1'].min()) / comparison_df['Test F1'].min() * 100
    print(f"  Improvement: {improvement:.1f}% (best vs worst)")

print("\nFiles generated:")
print("  1. model_comparison.png")
print("  2. model_comparison_results.csv")
