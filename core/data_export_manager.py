#!/usr/bin/env python3
"""
Data Export Manager for Multi-Dimensional Predictions
Handles Excel exports, database storage, and data visualization for fulfillment center analysis
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExportManager:
    """Manages data export, storage, and visualization for multi-dimensional predictions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.export_dir = self.data_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self.db_path = self.data_dir / "predictions.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # SKU predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sku_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    months_ahead INTEGER NOT NULL,
                    total_prediction REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    mape REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(sku, prediction_date, months_ahead)
                )
            """)
            
            # Channel breakdown table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    channel TEXT NOT NULL,
                    predicted_volume REAL NOT NULL,
                    percentage REAL NOT NULL,
                    approach TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sku, prediction_date) REFERENCES sku_predictions(sku, prediction_date)
                )
            """)
            
            # Geographical breakdown table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS geo_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    state TEXT NOT NULL,
                    predicted_volume REAL NOT NULL,
                    percentage REAL NOT NULL,
                    approach TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sku, prediction_date) REFERENCES sku_predictions(sku, prediction_date)
                )
            """)
            
            # Fulfillment center breakdown table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fulfillment_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    channel TEXT NOT NULL,
                    city TEXT NOT NULL,
                    godown_id TEXT NOT NULL,
                    predicted_volume REAL NOT NULL,
                    percentage REAL NOT NULL,
                    capacity_utilization REAL,
                    serves_states TEXT,
                    approach TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sku, prediction_date) REFERENCES sku_predictions(sku, prediction_date)
                )
            """)
            
            # Inventory allocation matrix table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventory_allocation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sku TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    channel TEXT NOT NULL,
                    state TEXT NOT NULL,
                    total_allocation REAL NOT NULL,
                    godown_city TEXT NOT NULL,
                    godown_allocation REAL NOT NULL,
                    allocation_percentage REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sku, prediction_date) REFERENCES sku_predictions(sku, prediction_date)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def store_prediction_results(self, sku: str, prediction_result: Dict):
        """Store complete prediction results in database"""
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        months_ahead = prediction_result.get('metadata', {}).get('months_ahead', 3)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store main SKU prediction
            cursor.execute("""
                INSERT OR REPLACE INTO sku_predictions 
                (sku, prediction_date, months_ahead, total_prediction, confidence_level, confidence_score, mape)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sku,
                prediction_date,
                months_ahead,
                prediction_result['total_prediction'],
                prediction_result.get('channel_breakdown', {}).get('confidence', 'Unknown'),
                0.5,  # Default confidence score
                None  # MAPE not available in multi-dimensional results
            ))
            
            # Store channel breakdown
            channel_breakdown = prediction_result.get('channel_breakdown', {})
            for channel, volume in channel_breakdown.get('predictions', {}).items():
                if volume > 0:
                    percentage = (volume / prediction_result['total_prediction'] * 100) if prediction_result['total_prediction'] > 0 else 0
                    cursor.execute("""
                        INSERT OR REPLACE INTO channel_predictions
                        (sku, prediction_date, channel, predicted_volume, percentage, approach, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sku, prediction_date, channel, volume, percentage,
                        channel_breakdown.get('approach', 'Unknown'),
                        channel_breakdown.get('confidence', 'Unknown')
                    ))
            
            # Store geographical breakdown
            geo_breakdown = prediction_result.get('geographical_breakdown', {})
            for state, volume in geo_breakdown.get('predictions', {}).items():
                if volume > 0:
                    percentage = (volume / prediction_result['total_prediction'] * 100) if prediction_result['total_prediction'] > 0 else 0
                    cursor.execute("""
                        INSERT OR REPLACE INTO geo_predictions
                        (sku, prediction_date, state, predicted_volume, percentage, approach, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sku, prediction_date, state, volume, percentage,
                        geo_breakdown.get('approach', 'Unknown'),
                        geo_breakdown.get('confidence', 'Unknown')
                    ))
            
            # Store fulfillment center breakdown
            godown_breakdown = prediction_result.get('godown_breakdown', {})
            for godown_key, volume in godown_breakdown.get('predictions', {}).items():
                if volume > 0 and '_' in godown_key:
                    channel, city = godown_key.split('_', 1)
                    percentage = (volume / prediction_result['total_prediction'] * 100) if prediction_result['total_prediction'] > 0 else 0
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO fulfillment_predictions
                        (sku, prediction_date, channel, city, godown_id, predicted_volume, percentage, 
                         capacity_utilization, serves_states, approach, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sku, prediction_date, channel, city, godown_key, volume, percentage,
                        None,  # Capacity utilization not in current result
                        None,  # Serves states not in current result
                        godown_breakdown.get('approach', 'Unknown'),
                        godown_breakdown.get('confidence', 'Unknown')
                    ))
            
            # Store inventory allocation matrix
            allocation_matrix = prediction_result.get('inventory_allocation_matrix', {})
            for allocation_key, allocation_data in allocation_matrix.items():
                if '_' in allocation_key:
                    channel, state = allocation_key.split('_', 1)
                    total_allocation = allocation_data['total_allocation']
                    
                    for godown_info in allocation_data.get('godowns', []):
                        cursor.execute("""
                            INSERT OR REPLACE INTO inventory_allocation
                            (sku, prediction_date, channel, state, total_allocation, 
                             godown_city, godown_allocation, allocation_percentage)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sku, prediction_date, channel, state, total_allocation,
                            godown_info['city'], godown_info['allocation'], godown_info['percentage']
                        ))
            
            conn.commit()
            logger.info(f"Stored prediction results for SKU: {sku}")
    
    def export_to_excel(self, prediction_results: Dict[str, Dict], filename: Optional[str] = None) -> Path:
        """Export prediction results to multi-sheet Excel file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fulfillment_predictions_{timestamp}.xlsx"
        
        excel_path = self.export_dir / filename
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary Overview
            summary_data = []
            for sku, result in prediction_results.items():
                summary_data.append({
                    'SKU': sku,
                    'Total_Prediction': result['total_prediction'],
                    'Channel_Confidence': result.get('channel_breakdown', {}).get('confidence', 'Unknown'),
                    'Geo_Confidence': result.get('geographical_breakdown', {}).get('confidence', 'Unknown'),
                    'Godown_Confidence': result.get('godown_breakdown', {}).get('confidence', 'Unknown'),
                    'Top_Channel': max(result.get('channel_breakdown', {}).get('predictions', {}).items(), 
                                     key=lambda x: x[1], default=('Unknown', 0))[0],
                    'Top_State': max(result.get('geographical_breakdown', {}).get('predictions', {}).items(),
                                   key=lambda x: x[1] if x[0] != 'Others' else 0, default=('Unknown', 0))[0],
                    'Analysis_Date': result.get('metadata', {}).get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Channel Breakdown
            channel_data = []
            for sku, result in prediction_results.items():
                channel_breakdown = result.get('channel_breakdown', {}).get('predictions', {})
                total_pred = result['total_prediction']
                for channel, volume in channel_breakdown.items():
                    if volume > 0:
                        channel_data.append({
                            'SKU': sku,
                            'Channel': channel,
                            'Predicted_Volume': volume,
                            'Percentage': (volume / total_pred * 100) if total_pred > 0 else 0,
                            'Approach': result.get('channel_breakdown', {}).get('approach', 'Unknown'),
                            'Confidence': result.get('channel_breakdown', {}).get('confidence', 'Unknown')
                        })
            
            channel_df = pd.DataFrame(channel_data)
            if not channel_df.empty:
                channel_df = channel_df.sort_values(['SKU', 'Predicted_Volume'], ascending=[True, False])
                channel_df.to_excel(writer, sheet_name='Channel_Breakdown', index=False)
            
            # Sheet 3: Geographical Breakdown
            geo_data = []
            for sku, result in prediction_results.items():
                geo_breakdown = result.get('geographical_breakdown', {}).get('predictions', {})
                total_pred = result['total_prediction']
                for state, volume in geo_breakdown.items():
                    if volume > 0:
                        geo_data.append({
                            'SKU': sku,
                            'State': state,
                            'Predicted_Volume': volume,
                            'Percentage': (volume / total_pred * 100) if total_pred > 0 else 0,
                            'Approach': result.get('geographical_breakdown', {}).get('approach', 'Unknown'),
                            'Confidence': result.get('geographical_breakdown', {}).get('confidence', 'Unknown')
                        })
            
            geo_df = pd.DataFrame(geo_data)
            if not geo_df.empty:
                geo_df = geo_df.sort_values(['SKU', 'Predicted_Volume'], ascending=[True, False])
                geo_df.to_excel(writer, sheet_name='Geographical_Breakdown', index=False)
            
            # Sheet 4: Fulfillment Center Breakdown
            fulfillment_data = []
            for sku, result in prediction_results.items():
                godown_breakdown = result.get('godown_breakdown', {}).get('predictions', {})
                total_pred = result['total_prediction']
                for godown_key, volume in godown_breakdown.items():
                    if volume > 0 and '_' in godown_key:
                        channel, city = godown_key.split('_', 1)
                        fulfillment_data.append({
                            'SKU': sku,
                            'Channel': channel,
                            'City': city,
                            'Godown_ID': godown_key,
                            'Predicted_Volume': volume,
                            'Percentage': (volume / total_pred * 100) if total_pred > 0 else 0,
                            'Approach': result.get('godown_breakdown', {}).get('approach', 'Unknown'),
                            'Confidence': result.get('godown_breakdown', {}).get('confidence', 'Unknown')
                        })
            
            fulfillment_df = pd.DataFrame(fulfillment_data)
            if not fulfillment_df.empty:
                fulfillment_df = fulfillment_df.sort_values(['Channel', 'City', 'Predicted_Volume'], 
                                                          ascending=[True, True, False])
                fulfillment_df.to_excel(writer, sheet_name='Fulfillment_Centers', index=False)
            
            # Sheet 5: Inventory Allocation Matrix
            allocation_data = []
            for sku, result in prediction_results.items():
                allocation_matrix = result.get('inventory_allocation_matrix', {})
                for allocation_key, allocation_info in allocation_matrix.items():
                    if '_' in allocation_key:
                        channel, state = allocation_key.split('_', 1)
                        total_allocation = allocation_info['total_allocation']
                        
                        for godown_info in allocation_info.get('godowns', []):
                            allocation_data.append({
                                'SKU': sku,
                                'Channel': channel,
                                'State': state,
                                'Total_State_Allocation': total_allocation,
                                'Godown_City': godown_info['city'],
                                'Godown_Allocation': godown_info['allocation'],
                                'Godown_Percentage_of_State': godown_info['percentage'],
                                'Godown_Percentage_of_Total': (godown_info['allocation'] / result['total_prediction'] * 100) 
                                                            if result['total_prediction'] > 0 else 0
                            })
            
            allocation_df = pd.DataFrame(allocation_data)
            if not allocation_df.empty:
                allocation_df = allocation_df.sort_values(['Channel', 'State', 'Godown_Allocation'], 
                                                        ascending=[True, True, False])
                allocation_df.to_excel(writer, sheet_name='Inventory_Allocation', index=False)
        
        logger.info(f"Excel export completed: {excel_path}")
        return excel_path
    
    def query_by_sku(self, sku: str, prediction_date: Optional[str] = None) -> Dict:
        """Query all predictions for a specific SKU"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            date_filter = f"AND prediction_date = '{prediction_date}'" if prediction_date else ""
            
            # Get main prediction
            cursor.execute(f"""
                SELECT * FROM sku_predictions 
                WHERE sku = ? {date_filter}
                ORDER BY prediction_date DESC LIMIT 1
            """, (sku,))
            main_result = cursor.fetchone()
            
            if not main_result:
                return {'error': f'No predictions found for SKU: {sku}'}
            
            pred_date = main_result['prediction_date']
            
            # Get channel breakdown
            cursor.execute("""
                SELECT * FROM channel_predictions 
                WHERE sku = ? AND prediction_date = ?
                ORDER BY predicted_volume DESC
            """, (sku, pred_date))
            channels = [dict(row) for row in cursor.fetchall()]
            
            # Get geographical breakdown
            cursor.execute("""
                SELECT * FROM geo_predictions 
                WHERE sku = ? AND prediction_date = ?
                ORDER BY predicted_volume DESC
            """, (sku, pred_date))
            geography = [dict(row) for row in cursor.fetchall()]
            
            # Get fulfillment centers
            cursor.execute("""
                SELECT * FROM fulfillment_predictions 
                WHERE sku = ? AND prediction_date = ?
                ORDER BY predicted_volume DESC
            """, (sku, pred_date))
            fulfillment = [dict(row) for row in cursor.fetchall()]
            
            # Get allocation matrix
            cursor.execute("""
                SELECT * FROM inventory_allocation 
                WHERE sku = ? AND prediction_date = ?
                ORDER BY channel, state, godown_allocation DESC
            """, (sku, pred_date))
            allocation = [dict(row) for row in cursor.fetchall()]
            
            return {
                'sku': sku,
                'prediction_info': dict(main_result),
                'channel_breakdown': channels,
                'geographical_breakdown': geography,
                'fulfillment_centers': fulfillment,
                'inventory_allocation': allocation
            }
    
    def query_by_fulfillment_center(self, channel: str, city: str, prediction_date: Optional[str] = None) -> Dict:
        """Query all SKUs for a specific fulfillment center"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            date_filter = f"AND prediction_date = '{prediction_date}'" if prediction_date else ""
            
            cursor.execute(f"""
                SELECT sku, predicted_volume, percentage, confidence, prediction_date
                FROM fulfillment_predictions 
                WHERE channel = ? AND city = ? {date_filter}
                ORDER BY prediction_date DESC, predicted_volume DESC
            """, (channel, city))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return {'error': f'No predictions found for {channel} in {city}'}
            
            total_volume = sum(row['predicted_volume'] for row in results)
            
            return {
                'fulfillment_center': f"{channel}_{city}",
                'channel': channel,
                'city': city,
                'total_predicted_volume': total_volume,
                'sku_count': len(results),
                'sku_predictions': results
            }
    
    def query_by_channel_state(self, channel: str, state: str, prediction_date: Optional[str] = None) -> Dict:
        """Query inventory allocation for specific channel-state combination"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            date_filter = f"AND prediction_date = '{prediction_date}'" if prediction_date else ""
            
            cursor.execute(f"""
                SELECT * FROM inventory_allocation 
                WHERE channel = ? AND state = ? {date_filter}
                ORDER BY prediction_date DESC, total_allocation DESC
            """, (channel, state))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results:
                return {'error': f'No allocation found for {channel} in {state}'}
            
            # Aggregate by godown
            godown_summary = {}
            for row in results:
                godown_city = row['godown_city']
                if godown_city not in godown_summary:
                    godown_summary[godown_city] = {
                        'total_allocation': 0,
                        'sku_count': 0,
                        'skus': []
                    }
                godown_summary[godown_city]['total_allocation'] += row['godown_allocation']
                godown_summary[godown_city]['sku_count'] += 1
                godown_summary[godown_city]['skus'].append({
                    'sku': row['sku'],
                    'allocation': row['godown_allocation']
                })
            
            return {
                'channel_state': f"{channel}_{state}",
                'channel': channel,
                'state': state,
                'total_skus': len(set(row['sku'] for row in results)),
                'godown_summary': godown_summary,
                'detailed_allocations': results
            }

def main():
    """Demo data export functionality"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    export_manager = DataExportManager(data_dir)
    
    print("🗄️  Data Export Manager - Ready")
    print(f"Database: {export_manager.db_path}")
    print(f"Export Directory: {export_manager.export_dir}")
    
    return export_manager

if __name__ == "__main__":
    export_manager = main()
