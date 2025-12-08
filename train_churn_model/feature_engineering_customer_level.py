import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Loading data...")
df = pd.read_csv('olist_customer_churn_data_customized.csv')
print(f"Original dataset: {df.shape}")
print(f"Orders: {len(df):,}")
print(f"Unique customers: {df['customer_unique_id'].nunique():,}")

initial_stats = {
    'total_orders': len(df),
    'unique_customers': df['customer_unique_id'].nunique(),
    'orders_per_customer': len(df) / df['customer_unique_id'].nunique()
}

print("\nPreparing temporal features...")
timestamp_cols = ['order_purchase_timestamp', 'order_approved_at',
                  'order_delivered_carrier_date', 'order_delivered_customer_date',
                  'order_estimated_delivery_date', 'review_creation_date']

for col in timestamp_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

if 'order_purchase_timestamp' in df.columns:
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['is_weekend_purchase'] = (df['purchase_day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['purchase_hour'] >= 9) & (df['purchase_hour'] <= 17)).astype(int)

print("\nCalculating reference date for recency...")
reference_date = df['order_purchase_timestamp'].max()
print(f"Reference date: {reference_date}")

print("\nCreating customer-level aggregations...")
customer_groups = df.groupby('customer_unique_id')
customer_features = {}

customer_features['customer_state'] = customer_groups['customer_state'].first()
customer_features['customer_city'] = customer_groups['customer_city'].first()
customer_features['customer_zip_code_prefix'] = customer_groups['customer_zip_code_prefix'].first()

customer_features['recency'] = customer_groups['order_purchase_timestamp'].apply(
    lambda x: (reference_date - x.max()).days if x.notna().any() else np.nan
)
customer_features['last_purchase_timestamp'] = customer_groups['order_purchase_timestamp'].max()
customer_features['frequency'] = customer_groups.size()
customer_features['monetary_total'] = customer_groups['price'].sum()
customer_features['monetary_mean'] = customer_groups['price'].mean()
customer_features['monetary_std'] = customer_groups['price'].std()
customer_features['monetary_max'] = customer_groups['price'].max()
customer_features['monetary_min'] = customer_groups['price'].min()

customer_features['avg_delivery_deviation'] = customer_groups['delivery_deviation'].mean()
customer_features['std_delivery_deviation'] = customer_groups['delivery_deviation'].std()
customer_features['total_late_deliveries'] = customer_groups.apply(
    lambda x: (x['delivery_deviation'] > 0).sum()
)
customer_features['late_delivery_rate'] = customer_groups.apply(
    lambda x: (x['delivery_deviation'] > 0).sum() / len(x)
)

customer_features['avg_shipping_time'] = customer_groups['shipping_time'].mean()
customer_features['avg_delivery_distance'] = customer_groups['delivery_distance'].mean()
customer_features['avg_freight_value'] = customer_groups['freight_value'].mean()
customer_features['total_freight_value'] = customer_groups['freight_value'].sum()
customer_features['avg_processing_time'] = customer_groups['processing_time'].mean()

customer_features['avg_days_between_orders'] = customer_groups['order_purchase_timestamp'].apply(
    lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else np.nan
)
customer_features['std_days_between_orders'] = customer_groups['order_purchase_timestamp'].apply(
    lambda x: x.sort_values().diff().dt.days.std() if len(x) > 1 else np.nan
)
customer_features['min_days_between_orders'] = customer_groups['order_purchase_timestamp'].apply(
    lambda x: x.sort_values().diff().dt.days.min() if len(x) > 1 else np.nan
)
customer_features['max_days_between_orders'] = customer_groups['order_purchase_timestamp'].apply(
    lambda x: x.sort_values().diff().dt.days.max() if len(x) > 1 else np.nan
)

customer_features['unique_products_purchased'] = customer_groups['product_id'].nunique()
customer_features['unique_categories_purchased'] = customer_groups['product_category_name'].nunique()
customer_features['most_frequent_category'] = customer_groups['product_category_name'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
)
customer_features['avg_product_weight'] = customer_groups['product_weight_g'].mean()
customer_features['avg_product_name_length'] = customer_groups['product_name_lenght'].mean()
customer_features['avg_product_photos'] = customer_groups['product_photos_qty'].mean()
customer_features['avg_product_complexity_score'] = customer_groups['product_complexity_score'].mean()

customer_features['avg_review_score'] = customer_groups['review_score'].mean()
customer_features['std_review_score'] = customer_groups['review_score'].std()
customer_features['min_review_score'] = customer_groups['review_score'].min()
customer_features['has_low_review'] = customer_groups['review_score'].apply(
    lambda x: (x <= 3).any() if x.notna().any() else 0
)

customer_features['most_frequent_payment_type'] = customer_groups['payment_type'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
)
customer_features['avg_payment_installments'] = customer_groups['payment_installments'].mean()
customer_features['max_payment_installments'] = customer_groups['payment_installments'].max()
customer_features['avg_payment_value'] = customer_groups['payment_value'].mean()
customer_features['total_payment_value'] = customer_groups['payment_value'].sum()

customer_features['most_frequent_order_status'] = customer_groups['order_status'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
)
customer_features['total_cross_state_deliveries'] = customer_groups['is_cross_state_delivery'].sum()
customer_features['cross_state_delivery_rate'] = customer_groups['is_cross_state_delivery'].mean()
customer_features['is_customer_from_capital'] = customer_groups['is_customer_capital'].first()
customer_features['capital_purchase_rate'] = customer_groups['is_customer_capital'].mean()

customer_features['most_common_purchase_hour'] = customer_groups['purchase_hour'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
)
customer_features['most_common_purchase_day'] = customer_groups['purchase_day_of_week'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
)
customer_features['weekend_purchase_rate'] = customer_groups['is_weekend_purchase'].mean()
customer_features['business_hours_purchase_rate'] = customer_groups['is_business_hours'].mean()

customer_features['avg_freight_to_price_ratio'] = customer_groups['freight_to_price_ratio'].mean()
customer_features['avg_price_deviation_from_category'] = customer_groups['price_deviation_from_category_median'].mean()
customer_features['avg_price_per_weight'] = customer_groups.apply(
    lambda x: (x['price'] / x['product_weight_g']).mean() if (x['product_weight_g'] > 0).any() else np.nan
)
customer_features['avg_delivery_distance_per_day'] = customer_groups.apply(
    lambda x: (x['delivery_distance'] / x['shipping_time']).mean() if (x['shipping_time'] > 0).any() else np.nan
)

customer_features['unique_sellers'] = customer_groups['seller_id'].nunique()
customer_features['avg_seller_delivery_deviation'] = customer_groups['seller_avg_delivery_deviation'].mean()
customer_features['most_frequent_seller_state'] = customer_groups['seller_state'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
)

customer_features['churn'] = customer_groups['churn'].max()

print("\nCombining features into customer-level dataset...")
customer_df = pd.DataFrame(customer_features)
customer_df.reset_index(inplace=True)
print(f"Customer-level dataset: {customer_df.shape}")

print("\nData quality checks...")
missing_summary = customer_df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(missing_summary) > 0:
    print(f"\nFeatures with missing values:")
    print(missing_summary.head(20))

churn_dist = customer_df['churn'].value_counts()
churn_rate = customer_df['churn'].mean()
print(f"\nChurn distribution:")
print(churn_dist)
print(f"Churn rate: {churn_rate:.2%}")

print("\nCreating visualizations...")
os.makedirs('feature_engineering_visualizations', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
order_counts = df.groupby('customer_unique_id').size()
ax.hist(order_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(order_counts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {order_counts.mean():.2f}')
ax.axvline(order_counts.median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {order_counts.median():.0f}')
ax.set_xlabel('Number of Orders per Customer', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.set_title('Distribution of Orders per Customer', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
churn_counts = customer_df['churn'].value_counts()
colors = ['#66BB6A', '#EF5350']
bars = ax.bar(['No Churn', 'Churn'], churn_counts.values, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.set_title(f'Churn Distribution (Rate: {churn_rate:.2%})', fontsize=13)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(customer_df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11)

ax = axes[1, 0]
numerical_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if 'id' not in col.lower() and col != 'customer_zip_code_prefix']
correlations = customer_df[numerical_cols].corrwith(customer_df['churn']).abs().sort_values(ascending=False)
top_20_corr = correlations.head(20)
colors = ['#EF5350' if val > 0.1 else '#42A5F5' for val in top_20_corr.values]
bars = ax.barh(range(len(top_20_corr)), top_20_corr.values, color=colors, edgecolor='black', linewidth=1)
ax.set_yticks(range(len(top_20_corr)))
ax.set_yticklabels([col.replace('_', ' ').title() for col in top_20_corr.index], fontsize=9)
ax.set_xlabel('Correlation with Churn', fontsize=12)
ax.set_title('Top 20 Features by Correlation', fontsize=13)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

ax = axes[1, 1]
ax.axis('off')
summary_text = f"""Transformation Summary

Input (Order-Level):
  Total Orders: {initial_stats['total_orders']:,}
  Unique Customers: {initial_stats['unique_customers']:,}

Output (Customer-Level):
  Total Customers: {len(customer_df):,}
  Total Features: {len(customer_df.columns):,}
  Churn Rate: {churn_rate:.2%}
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace')

plt.tight_layout()
plt.savefig('feature_engineering_visualizations/transformation_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaving customer-level dataset...")
output_file = 'customer_level_features.csv'
customer_df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")

feature_list = [col for col in customer_df.columns if col not in ['customer_unique_id', 'churn']]
pd.DataFrame({'feature': feature_list}).to_csv('customer_level_feature_list.csv', index=False)

print(f"\nTransformation complete:")
print(f"  Original: {df.shape[0]:,} orders")
print(f"  New: {customer_df.shape[0]:,} customers")
print(f"  Features: {len(feature_list)}")
print(f"  Churn rate: {churn_rate:.2%}")
