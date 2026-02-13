
import pandas as pd
import numpy as np
import pyodbc
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import logging

# Setup Logging
logging.basicConfig(
    filename='db_model_test_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_DIR = Path("D:/software/ecom_analysis/data")
PROCESSED_DIR = DATA_DIR / "processed"

warnings.filterwarnings('ignore')

class DBModelTester:
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.raw_data = None
        self.segment_a = None  # 2019 - Oct 2024
        self.segment_c = None  # Jan 2025 - Present
        self.seasonal_multipliers = {}
        self.sku_mapping = {}
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        self.load_sku_mapping()

    def load_sku_mapping(self):
        try:
            sku_file = DATA_DIR / "raw" / "sku_list.csv"
            if sku_file.exists():
                sku_df = pd.read_csv(sku_file)
                # Assuming columns 'sku' and 'category'
                if 'sku' in sku_df.columns and 'category' in sku_df.columns:
                    self.sku_mapping = dict(zip(sku_df['sku'], sku_df['category']))
                logging.info(f"Loaded {len(self.sku_mapping)} SKUs from mapping file.")
            else:
                logging.warning("SKU mapping file not found.")
        except Exception as e:
            logging.error(f"Error loading SKU mapping: {e}")

    def fetch_data(self):
        cache_file = DATA_DIR / "raw" / "db_sales_cache.csv"
        
        if cache_file.exists():
            logging.info(f"Loading data from local cache: {cache_file}")
            print(f"Loading data from local cache: {cache_file}")
            self.raw_data = pd.read_csv(cache_file)
            
            # Ensure Date parsing on load
            if 'Date' in self.raw_data.columns:
                self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        else:
            logging.info("Connecting to database...")
            print("Connecting to database...")
            try:
                conn = pyodbc.connect(self.conn_str)
                query = "SELECT * FROM SALESTRANSACTIONS"
                logging.info(f"Executing query: {query}")
                print("Executing query...")
                self.raw_data = pd.read_sql(query, conn)
                conn.close()
                
                # Save to cache
                DATA_DIR.joinpath("raw").mkdir(exist_ok=True)
                self.raw_data.to_csv(cache_file, index=False)
                logging.info(f"Saved data to local cache: {cache_file}")
                print(f"Saved data to local cache: {cache_file}")
                
            except Exception as e:
                logging.error(f"Error fetching data: {e}")
                raise e

        # Debugging Columns
        logging.info(f"Raw Data Columns: {self.raw_data.columns.tolist()}")
        print(f"Raw Data Columns: {self.raw_data.columns.tolist()}")
        
        # Normalize Columns (Handle variations)
        cols = {c.lower(): c for c in self.raw_data.columns}
        
        # Date mappings
        if 'transactiondate' in cols:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data[cols['transactiondate']])
        elif 'date' in cols:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data[cols['date']])
        elif 'orderdate' in cols:
            self.raw_data['Date'] = pd.to_datetime(self.raw_data[cols['orderdate']])
        else:
            logging.error("Date column not found! Available columns: " + str(list(cols.keys())))
            print("ERROR: Date column not found!")
            print("Available columns:", self.raw_data.columns.tolist())
            print("First row sample:", self.raw_data.iloc[0].to_dict() if not self.raw_data.empty else "Empty DataFrame")
            raise KeyError("Date")
        
        # SKU/Product mappings
        if 'skuid' in cols:
             self.raw_data['SKU'] = self.raw_data[cols['skuid']]
        elif 'sku' in cols:
             self.raw_data['SKU'] = self.raw_data[cols['sku']]
             
        # Quantity mappings
        if 'quantity' in cols:
             self.raw_data['Quantity'] = self.raw_data[cols['quantity']]
        elif 'qty' in cols:
             self.raw_data['Quantity'] = self.raw_data[cols['qty']]
             
        logging.info(f"Data fetched/loaded: {len(self.raw_data)} records")
        
        # Map Categories if not present
        if 'Category' not in self.raw_data.columns and self.sku_mapping:
            self.raw_data['Category'] = self.raw_data['SKU'].map(self.sku_mapping)
            
        logging.info(f"Columns after processing: {self.raw_data.columns.tolist()}")

    def prepare_segments(self):
        df = self.raw_data.copy()
        df = df.sort_values('Date')
        
        # Segment A: 2019 - Oct 2024
        self.segment_a = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2024-10-31')]
        
        # Segment C: Jan 2025 - Present
        self.segment_c = df[df['Date'] >= '2025-01-01']
        
        logging.info(f"Segment A (Historical): {len(self.segment_a)} records")
        logging.info(f"Segment C (Current): {len(self.segment_c)} records")

    def calculate_seasonal_multipliers(self):
        logging.info("Calculating seasonal multipliers...")
        if 'Category' not in self.segment_a.columns:
            logging.warning("Category column missing, skipping seasonal multipliers.")
            return

        self.segment_a['Month'] = self.segment_a['Date'].dt.month
        
        # Average Monthly Quantity per Category
        cat_month_avg = self.segment_a.groupby(['Category', 'Month'])['Quantity'].mean()
        cat_overall_avg = self.segment_a.groupby('Category')['Quantity'].mean()
        
        for category in cat_overall_avg.index:
            overall = cat_overall_avg[category]
            if overall == 0: continue
            
            self.seasonal_multipliers[category] = {}
            for month in range(1, 13):
                if (category, month) in cat_month_avg.index:
                    multiplier = cat_month_avg[(category, month)] / overall
                    self.seasonal_multipliers[category][month] = multiplier
                else:
                    self.seasonal_multipliers[category][month] = 1.0
                    
        logging.info("Seasonal multipliers calculated.")

    def create_features(self, df):
        # Aggregate to Daily-SKU level if not already
        if len(df) == 0: return df
        
        daily = df.groupby(['Date', 'SKU', 'Category']).agg({
            'Quantity': 'sum'
        }).reset_index() if 'Category' in df.columns else df.groupby(['Date', 'SKU']).agg({'Quantity': 'sum'}).reset_index()
        
        daily['Month'] = daily['Date'].dt.month
        daily['Day'] = daily['Date'].dt.day
        daily['IsPayday'] = daily['Day'].isin([1, 15]).astype(int)
        daily['DaysSinceRestart'] = (daily['Date'] - pd.to_datetime('2025-01-01')).dt.days
        
        # Add Seasonal Multiplier
        if 'Category' in daily.columns:
            daily['SeasonalMultiplier'] = daily.apply(
                lambda x: self.seasonal_multipliers.get(x['Category'], {}).get(x['Month'], 1.0), axis=1
            )
        else:
            daily['SeasonalMultiplier'] = 1.0
            
        # Lags and Rolling
        # Create lags for each SKU (requires sorting)
        daily = daily.sort_values(['SKU', 'Date'])
        
        for lag in [1, 7, 30]:
            daily[f'Lag_{lag}'] = daily.groupby('SKU')['Quantity'].shift(lag)
            
        for window in [7, 30]:
            daily[f'RollingMean_{window}'] = daily.groupby('SKU')['Quantity'].transform(lambda x: x.rolling(window).mean())
        
        # Drop NaNs created by lags
        daily = daily.dropna()
        
        return daily

    def backtest_chunk_size(self):
        logging.info("Starting Chunk Size Sensitivity Test...")
        
        # Only use Segment C for Train/Test in Walk-Forward
        # Train on [Jan..Month-1], Predict [Month].
        
        # Check available months in Segment C
        self.segment_c['Month'] = self.segment_c['Date'].dt.month
        months = sorted(self.segment_c['Month'].unique())
        
        results = []
        
        # Walk forward from July (7) to Dec (12)
        # Assuming we have data up to Dec? Or we simulate?
        # User says: "Train Jan-June 2025 -> Predict July"
        
        all_results_data = []
        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        
        for predict_month in range(7, 13):
            if predict_month not in months:
                logging.info(f"Skipping month {predict_month}, no data.")
                continue
                
            test_data_raw = self.segment_c[self.segment_c['Month'] == predict_month]
            if len(test_data_raw) == 0: continue
            
            # Test different lookback windows (Chunk Sizes)
            # 5, 6, 7, 8, 9 months lookback.
            # Max lookback is predict_month - 1 (e.g., for July (7), max lookback is 6 months (Jan-Jun))
            max_lookback = predict_month - 1
            possible_windows = [w for w in [5, 6, 7, 8, 9] if w <= max_lookback]
            
            for window in possible_windows:
                start_month = predict_month - window
                
                # Format Training Range String (e.g., "Jan-Jun")
                range_str = f"{month_names[start_month]}-{month_names[predict_month-1]}"
                
                # print(f"Processing {month_names[predict_month]} (Train: {range_str}, Window: {window})...", end='\r') # Reduced spam
                
                # Prepare Train
                train_data_raw = self.segment_c[
                    (self.segment_c['Month'] >= start_month) & 
                    (self.segment_c['Month'] < predict_month)
                ]
                
                # Feature Engineering
                # We need to process train and test together to handle Lags correctly across boundaries.
                # So we should pass the full context but mask for training.
                
                full_window = pd.concat([train_data_raw, test_data_raw])
                features = self.create_features(full_window)
                
                # Split back
                train_features = features[features['Month'] < predict_month]
                test_features = features[features['Month'] == predict_month]
                
                if len(train_features) < 10 or len(test_features) < 1:
                    logging.warning(f"Insufficient data for window {window}")
                    continue
                
                # Train
                feature_cols = [c for c in train_features.columns if c not in ['Date', 'SKU', 'Category', 'Quantity', 'Month', 'Day']]
                
                X_train = train_features[feature_cols]
                y_train = train_features['Quantity']
                X_test = test_features[feature_cols]
                y_test = test_features['Quantity']
                
                self.model.fit(self.scaler.fit_transform(X_train), y_train)
                
                # Predict
                preds = self.model.predict(self.scaler.transform(X_test))
                
                # Sum for Absolute Numbers context
                mape_overall = mean_absolute_percentage_error(y_test, preds)
                total_actual = y_test.sum()
                total_predicted = preds.sum()
                
                # Weighted MAPE (wMAPE) - Better for "Aggregate" accuracy
                # Sum(|Actual - Predicted|) / Sum(Actual)
                abs_error_sum = np.sum(np.abs(y_test - preds))
                wmape = abs_error_sum / total_actual if total_actual > 0 else 0
                
                results.append({
                    'PredictMonth': predict_month,
                    'LookbackWindow': window,
                    'MAPE': mape_overall,
                    'wMAPE': wmape,
                    'Accuracy': max(0, 1 - wmape) * 100, # Using wMAPE for "Volume Accuracy"
                    'TotalActual': total_actual,
                    'TotalPredicted': total_predicted
                })
                
                # ---------------------------------------------------------
                # SKU-Level Aggregation (DAILY -> MONTHLY)
                # ---------------------------------------------------------
                # test_features is DAILY. We must aggregate to Month for report.
                daily_res = test_features[['SKU']].copy()
                if 'Category' in test_features.columns:
                    daily_res['Category'] = test_features['Category']
                else:
                    daily_res['Category'] = 'Unknown'
                    
                daily_res['Actual'] = y_test
                daily_res['Predicted'] = preds
                
                # Group by SKU to get Monthly Totals
                sku_monthly = daily_res.groupby(['SKU', 'Category']).agg({
                    'Actual': 'sum',
                    'Predicted': 'sum'
                }).reset_index()
                
                sku_monthly['Diff'] = sku_monthly['Predicted'] - sku_monthly['Actual']
                sku_monthly['AbsError'] = sku_monthly['Diff'].abs()
                
                # Calculate Monthly Metrics per SKU
                sku_monthly['SKU_MAPE'] = (sku_monthly['AbsError'] / sku_monthly['Actual']).replace([np.inf, -np.inf], 0)
                sku_monthly['SKU_Accuracy'] = (1 - sku_monthly['SKU_MAPE']).clip(lower=0)
                
                # Metadata
                sku_monthly['MonthNum'] = predict_month
                sku_monthly['Month'] = month_names[predict_month]
                sku_monthly['Window'] = window
                sku_monthly['TrainRange'] = range_str
                
                all_results_data.append(sku_monthly)
                
        results_df = pd.DataFrame(results)
        
        # Generate Consolidated Master Report
        if all_results_data:
            print("\nGenerating Consolidated Report...")
            full_sku_df = pd.concat(all_results_data)
            
            # Pivot Strategy:
            # Index: [SKU, MonthNum, Month, Actual]
            
            # 1. Base Info: SKU, Category, MonthNum, Month, Actual
            base_info = full_sku_df.groupby(['SKU', 'Category', 'MonthNum', 'Month'])['Actual'].max().reset_index()
            
            # 2. Pivot Metrics
            pivot_df = full_sku_df.pivot_table(
                index=['SKU', 'MonthNum'],
                columns='Window',
                values=['Predicted', 'SKU_Accuracy', 'SKU_MAPE', 'TrainRange'],
                aggfunc='first'
            )
            
            # 3. Construct Final DataFrame with Grouped Columns
            # We want: SKU, Month... then W5 [Range, Act, Pred, Acc, MAPE]... W6 [...]
            
            # Start with base info
            final_df = base_info.copy()
            
            # Available windows
            available_windows = sorted([c for c in full_sku_df['Window'].unique()])
            
            for w in available_windows:
                # Extract metrics for this window from pivot
                # Pivot has MultiIndex columns: (Metric, Window)
                
                # Range
                if ('TrainRange', w) in pivot_df.columns:
                    final_df[f'W{w}_Range'] = base_info.merge(pivot_df[('TrainRange', w)], on=['SKU', 'MonthNum'], how='left')[('TrainRange', w)]
                
                # Actual (Redundant but requested side-by-side)
                final_df[f'W{w}_Actual'] = final_df['Actual']
                
                # Predicted
                if ('Predicted', w) in pivot_df.columns:
                    final_df[f'W{w}_Predicted'] = base_info.merge(pivot_df[('Predicted', w)], on=['SKU', 'MonthNum'], how='left')[('Predicted', w)]
                
                # Accuracy
                if ('SKU_Accuracy', w) in pivot_df.columns:
                    final_df[f'W{w}_Accuracy'] = base_info.merge(pivot_df[('SKU_Accuracy', w)], on=['SKU', 'MonthNum'], how='left')[('SKU_Accuracy', w)]
                
                # MAPE
                if ('SKU_MAPE', w) in pivot_df.columns:
                    final_df[f'W{w}_MAPE'] = base_info.merge(pivot_df[('SKU_MAPE', w)], on=['SKU', 'MonthNum'], how='left')[('SKU_MAPE', w)]

            # Sort
            final_df = final_df.sort_values(['MonthNum', 'Actual'], ascending=[True, False])
            
            # column cleanup
            cols_static = ['SKU', 'Category', 'Month'] # Removing 'Actual' from start to avoid clutter/confusion vs grouped actuals
            cols_dynamic = [c for c in final_df.columns if c.startswith('W') and c[1].isdigit()]
            
            # Ensure dynamic cols are sorted by Window then by logical order
            def sort_key_win(c):
                # c looks like "W5_Pred"
                try:
                    parts = c.split('_')
                    win_num = int(parts[0][1:])
                    # Order: Range, Actual, Pred, Acc, MAPE
                    suffix = parts[1]
                    order_map = {'Range': 0, 'Actual': 1, 'Predicted': 2, 'Accuracy': 3, 'MAPE': 4}
                    return win_num * 10 + order_map.get(suffix, 9)
                except: return 999
            
            cols_dynamic.sort(key=sort_key_win)
            
            final_df = final_df[cols_static + cols_dynamic]
            
            # Save
            outfile = PROCESSED_DIR / "Final_Consolidated_All_Chunks_Report.csv"
            final_df.to_csv(outfile, index=False)
            
            print(f"Final Report: {outfile}")
            
        return results_df

    def save_visualization_data(self, comparison_df, month, window):
        # This function is no longer called directly from backtest_chunk_size
        # The consolidated report replaces its primary function.
        # Keeping it for potential future use or if other parts of the code call it.
        
        # Aggregate by SKU (and Category if exists)
        group_cols = ['SKU', 'Category'] if 'Category' in comparison_df.columns else ['SKU']
        
        sku_res = comparison_df.groupby(group_cols).agg({
            'Actual': 'sum', 
            'Predicted': 'sum'
        }).reset_index()
        
        sku_res['Diff'] = sku_res['Predicted'] - sku_res['Actual']
        sku_res['AbsError'] = abs(sku_res['Diff'])
        
        # Avoid division by zero
        sku_res['MAPE'] = (sku_res['AbsError'] / sku_res['Actual']).replace([np.inf, -np.inf], 0)
        
        # Accuracy = 1 - MAPE (floored at 0)
        sku_res['Accuracy'] = (1 - sku_res['MAPE']).clip(lower=0)
        
        # Convert to percentage for readability in CSV? Or keep as decimal?
        # User asked for "accuracy", usually %. Let's make them explicit.
        sku_res['MAPE_Pct'] = sku_res['MAPE'] * 100
        sku_res['Accuracy_Pct'] = sku_res['Accuracy'] * 100
        
        # Sort by Error (descending) to show worst performers first, or by Volume?
        # Usually high volume items matter more. Let's sort by Actual Volume Desc.
        sku_res = sku_res.sort_values('Actual', ascending=False)
        
        # Save to CSV
        filename = f"sku_report_month_{month}_lookback_{window}.csv"
        output_file = PROCESSED_DIR / filename
        sku_res.to_csv(output_file, index=False)
        # logging.info(f"Saved detailed SKU report: {filename}") # Reduce log spam?
        
        # Only plot for a specific window to avoid spamming images? 
        # Or just overwrite "latest"? Let's skip plotting for every single one to save time/space unless needed.
        # We can plot if it's the "Best" window, but we don't know yet.
        # Let's just save CSVs as requested.

if __name__ == "__main__":
    conn = "Driver={ODBC Driver 17 for SQL Server};Server=103.87.173.236,1433;Database=Tenant1_SalesDB;UID=apsr;PWD=Apsr@389;Encrypt=no;TrustServerCertificate=yes;Connection Timeout=120;"
    
    tester = DBModelTester(conn)
    try:
        tester.fetch_data()
        tester.prepare_segments()
        tester.calculate_seasonal_multipliers()
        results = tester.backtest_chunk_size()
        
        if not results.empty:
            logging.info("\nBacktest Results:")
            logging.info(results.to_string())
            
            # Formatting Output
            print("\n=== Backtest Validation Results ===")
            print(f"{'Month':<6} | {'Lookback':<9} | {'wMAPE':<8} | {'MAPE':<8} | {'Accuracy':<8} | {'Actual Qty':<12} | {'Pred Qty':<12}")
            print("-" * 88)
            
            for _, row in results.iterrows():
                # Accuracy here is based on wMAPE (Volume Accuracy)
                print(f"{int(row['PredictMonth']):<6} | {int(row['LookbackWindow']):<9} | {row['wMAPE']:.1%}   | {row['MAPE']:.1%}   | {row['Accuracy']:.1f}%    | {int(row['TotalActual']):<12} | {int(row['TotalPredicted']):<12}")
            
            print("-" * 88)
            
            # Find best window based on wMAPE
            best_run = results.loc[results['wMAPE'].idxmin()]
            best_month = int(best_run['PredictMonth'])
            best_window = int(best_run['LookbackWindow'])
            best_wmape = best_run['wMAPE']
            
            logging.info(f"Best Configuration: Month {best_month}, Lookback {best_window} months")
            print(f"\n>> Best Configuration: Month {best_month}, Lookback {best_window}")
            print(f">> Lowest wMAPE: {best_wmape:.1%} (i.e., Accuracy {best_run['Accuracy']:.1f}%)")
            
            # Detailed Analysis of the Best Run
            best_file = PROCESSED_DIR / f"sku_report_month_{best_month}_lookback_{best_window}.csv"
            if best_file.exists():
                print(f"\n=== Detailed Analysis (saved to {best_file}) ===")
                detail_df = pd.read_csv(best_file)
                
                # Total Stats
                total_act = detail_df['Actual'].sum()
                total_pred = detail_df['Predicted'].sum()
                net_diff = total_pred - total_act
                net_diff_pct = (net_diff / total_act) * 100
                
                print(f"Total Actual:    {total_act:,.0f}")
                print(f"Total Predicted: {total_pred:,.0f}")
                print(f"Net Difference:  {net_diff:+,.0f} ({net_diff_pct:+.1f}%)")
                print(f"Weighted Error:  {int(detail_df['AbsError'].sum()):,.0f} units (Sum of all individual mistakes)")
                print(f"\nMake sense? The Net Difference is small ({net_diff_pct:+.1f}%), but the Weighted Error is higher ({best_wmape:.1%}).")
                print("This means the model predicts the right *amount* of stock, but for the *wrong* products sometimes.")
                
                print("\n--- TOP 5 BIGGEST MISSES (By Volume) ---")
                print(f"{'SKU':<15} | {'Actual':<8} | {'Pred':<8} | {'Diff':<8} | {'Error Contribution'}")
                detail_df['Contribution'] = detail_df['AbsError'] / detail_df['AbsError'].sum()
                for _, row in detail_df.sort_values('AbsError', ascending=False).head(5).iterrows():
                    print(f"{str(row['SKU'])[:15]:<15} | {int(row['Actual']):<8} | {int(row['Predicted']):<8} | {int(row['Diff']):<8} | {row['Contribution']:.1%}")
            
            print(f"\n>> Full SKU-level reports saved in: {PROCESSED_DIR}")
            
        else:
            logging.info("No results generated.")
            print("No results generated. Check log for details.")
            
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        print(f"Failed: {e}")

