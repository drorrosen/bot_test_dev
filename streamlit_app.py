"""
Streamlit Chatbot for Retail Data Analysis
Using proven working pattern from pl_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime
import io

# # Import our retail analyzer # WE WILL NOT USE THIS FOR DATA LOADING ANYMORE
# from retail_analyzer import RetailAnalyzer 


def find_header_row(df_peek: pd.DataFrame, max_rows_to_check=10) -> int:
    """Find the most likely header row in the first few rows of a DataFrame."""
    for i in range(min(max_rows_to_check, len(df_peek))):
        row_values = df_peek.iloc[i]
        # A good header row has mostly non-null, string-like values
        string_like_count = 0
        non_null_count = 0
        for val in row_values:
            if pd.notna(val):
                non_null_count += 1
                if isinstance(val, str) and len(val.strip()) > 1 and not val.strip().replace('.', '', 1).isdigit():
                    string_like_count += 1
        
        # If a good proportion of cells are non-null and string-like, assume it's the header
        if non_null_count > len(df_peek.columns) * 0.6 and string_like_count > non_null_count * 0.5:
            return i
    return 0 # Default to first row if no better one is found

def load_and_process_data_directly(uploaded_file):
    """Load and process data directly from uploaded file, bypassing RetailAnalyzer for data loading."""
    temp_path = None
    try:
        temp_dir = tempfile.gettempdir()
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_path = os.path.join(temp_dir, f"streamlit_temp_{int(time.time())}{file_extension}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load a few rows to detect header, without setting header yet
        df_peek = pd.read_excel(temp_path, header=None, nrows=15) 
        
        # Remove completely empty top rows before header detection
        first_valid_index = 0
        for i in range(len(df_peek)):
            if df_peek.iloc[i].notna().any():
                first_valid_index = i
                break
        df_peek_content = df_peek.iloc[first_valid_index:]

        header_row_index_in_peek = find_header_row(df_peek_content)
        actual_header_row_in_file = first_valid_index + header_row_index_in_peek
        
        # Now load the full sheet using the detected header row
        data = pd.read_excel(temp_path, header=actual_header_row_in_file)
        
        # Basic cleaning: remove fully empty rows/cols
        data.dropna(axis=0, how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        
        # Clean column names (simple strip)
        data.columns = [str(col).strip() for col in data.columns]

        st.session_state.data = data
        st.session_state.results = {} # Reset results as RetailAnalyzer is not used for main processing
        
        # Clear any cached YoY summary so it recalculates with new data
        if 'yoy_summary' in st.session_state:
            del st.session_state.yoy_summary

        return True, "✅ Data loaded and preprocessed directly!"
        
    except Exception as e:
        st.error(f"❌ Failed to load/process data directly: {e}")
        return False, f"❌ Failed to load/process data: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                time.sleep(0.1)
                os.remove(temp_path)
            except (OSError, PermissionError):
                pass


def get_data_summary():
    """Get summary of analyzed data for the chatbot - CALCULATE YoY OURSELVES!"""
    if 'data' not in st.session_state or st.session_state.data is None:
        return None
    
    # Check if we already calculated this summary
    if 'yoy_summary' in st.session_state:
        return st.session_state.yoy_summary
    
    data = st.session_state.data
    
    # Start with basic info
    summary = {
        "dataset_info": {
            "total_records": len(data),
            "total_columns": len(data.columns),
            "columns": list(data.columns)
        },
        "yoy_calculations": {},
        "product_performance": {},
        "category_analysis": {}
    }
    
    # FORCE YoY CALCULATIONS - Simple and Direct
    # Check if we have the required columns
    required_data_cols = ['Metric', '52W_TY', '52W_LY', '12W_TY', '12W_LY', '4W_TY', '4W_LY']
    if not all(col in data.columns for col in required_data_cols):
        summary['yoy_calculations'] = {"error": "Missing required columns for YoY calculation (needs Metric, and all TY/LY pairs)."}
        st.session_state.yoy_summary = summary # Cache the error state
        return summary
        
    yoy_results = {}
    primary_metric_to_sum = 'Sales Value (£)'

    # --- Helper to find and process existing YoY columns ---
    def get_existing_yoy_average(period_str_simple): 
        period_prefix = period_str_simple.split('W')[0] + 'W' 
        if period_str_simple.startswith("52"):
            search_terms = [f'{period_prefix.lower()}', '52 w/e change', '52w yoy', '52 w/e chg']
        elif period_str_simple.startswith("12"):
            search_terms = [f'{period_prefix.lower()}', '12 w/e change', '12w yoy', '12 w/e chg']
        elif period_str_simple.startswith("4"):
            search_terms = [f'{period_prefix.lower()}', '4 w/e change', '4w yoy', '4 w/e chg']
        else:
            return None, None

        for col_name in data.columns:
            col_lower = col_name.lower()
            is_potential_yoy = any(term in col_lower for term in search_terms) and \
                                 ("change" in col_lower or "yoy" in col_lower or "chg" in col_lower) and \
                                 "ty" not in col_lower and "ly" not in col_lower
            if is_potential_yoy:
                try:
                    existing_col_numeric = pd.to_numeric(data[col_name], errors='coerce')
                    # Drop NaNs before calculating mean to avoid issues if all are NaN
                    existing_col_numeric_no_na = existing_col_numeric.dropna()
                    if existing_col_numeric_no_na.empty:
                        continue # Skip if column becomes empty after dropping NaNs
                        
                    existing_col_avg = existing_col_numeric_no_na.mean()
                    
                    if pd.notna(existing_col_avg):
                        # Heuristic: If avg is a small decimal (e.g. abs(avg) < 1.5, typical for decimal %),
                        # assume it needs to be multiplied by 100 to become a percentage value.
                        # This handles cases like 0.0014 (0.14%) vs 0.14 (14%).
                        # If it's already a large number (e.g. 14 for 14%), it might be an error to multiply.
                        # We assume existing columns are either direct percentages or decimals needing x100.
                        # A more robust solution would require knowing the exact format of these columns.
                        # For now, let's assume if its abs value is < 1.5 it's a decimal representation of percentage.
                        final_avg_for_display = existing_col_avg * 100 # Always multiply by 100 as per user insight
                        
                        return col_name, round(final_avg_for_display, 2)
                except Exception as e:
                    pass
        return None, None

    # --- Period Calculations (52W, 12W, 4W) ---
    for period_label, ty_col_name, ly_col_name in [("52W", "52W_TY", "52W_LY"), 
                                                     ("12W", "12W_TY", "12W_LY"), 
                                                     ("4W", "4W_TY", "4W_LY")]:
        if ty_col_name in data.columns and ly_col_name in data.columns:
            current_period_results = {}

            # 1. Overall YoY based on SUM of primary metric (e.g., 'Sales Value (£)')
            df_period_primary_metric = data[data['Metric'] == primary_metric_to_sum].copy()
            total_ty = pd.to_numeric(df_period_primary_metric[ty_col_name], errors='coerce').fillna(0).sum()
            total_ly = pd.to_numeric(df_period_primary_metric[ly_col_name], errors='coerce').fillna(0).sum()
            
            current_period_results["total_primary_metric_ty"] = round(total_ty, 2)
            current_period_results["total_primary_metric_ly"] = round(total_ly, 2)
            current_period_results["overall_growth_primary_metric"] = round(((total_ty - total_ly) / total_ly * 100) if total_ly != 0 else 0, 2)
            current_period_results["primary_metric_for_overall"] = primary_metric_to_sum

            # 2. Average of Row-wise YoY for ALL numeric rows
            yoy_all_numeric_rows = []
            for index, row in data.iterrows():
                ty_val_str = row.get(ty_col_name)
                ly_val_str = row.get(ly_col_name)
                ty_val_num = pd.to_numeric(ty_val_str, errors='coerce')
                ly_val_num = pd.to_numeric(ly_val_str, errors='coerce')
                if pd.notna(ty_val_num) and pd.notna(ly_val_num):
                    if ly_val_num != 0:
                        yoy_all_numeric_rows.append(((ty_val_num - ly_val_num) / ly_val_num) * 100)
                    else:
                        yoy_all_numeric_rows.append(0) 
            
            yoy_series_all_numeric = pd.Series(yoy_all_numeric_rows).dropna()
            
            if not yoy_series_all_numeric.empty:
                current_period_results["average_of_row_wise_growth_all_numeric"] = round(yoy_series_all_numeric.mean(), 2)
                current_period_results["std_dev_of_row_wise_growth_all_numeric"] = round(yoy_series_all_numeric.std(), 2)
                current_period_results["min_row_wise_growth_all_numeric"] = round(yoy_series_all_numeric.min(), 2)
                current_period_results["max_row_wise_growth_all_numeric"] = round(yoy_series_all_numeric.max(), 2)
                current_period_results["positive_rows_count_all_numeric"] = int((yoy_series_all_numeric > 0).sum())
                current_period_results["negative_rows_count_all_numeric"] = int((yoy_series_all_numeric < 0).sum())
                current_period_results["total_numeric_rows_for_avg"] = len(yoy_series_all_numeric)
            else:
                current_period_results["average_of_row_wise_growth_all_numeric"] = None
                current_period_results["std_dev_of_row_wise_growth_all_numeric"] = None

            # 3. Average from existing column (comparison)
            existing_col_name, existing_avg = get_existing_yoy_average(period_label)
            current_period_results["average_from_existing_column"] = existing_avg
            current_period_results["existing_column_name_used"] = existing_col_name
            current_period_results["calculation_type"] = f"{ty_col_name} vs {ly_col_name}"

            yoy_results[f'{period_label}_YoY'] = current_period_results
        else:
            pass

    summary["yoy_calculations"] = yoy_results
    
    # --- Top/Bottom Absolute Change Drivers for Primary Metric (52W) ---
    if ('Product Name' in data.columns and 
        'Metric' in data.columns and 
        '52W_TY' in data.columns and '52W_LY' in data.columns and 
        primary_metric_to_sum in data['Metric'].unique()):
        
        df_primary_metric = data[data['Metric'] == primary_metric_to_sum].copy()
        df_primary_metric['52W_TY_num'] = pd.to_numeric(df_primary_metric['52W_TY'], errors='coerce').fillna(0)
        df_primary_metric['52W_LY_num'] = pd.to_numeric(df_primary_metric['52W_LY'], errors='coerce').fillna(0)
        
        product_sales = df_primary_metric.groupby('Product Name').agg(
            total_ty = ('52W_TY_num', 'sum'),
            total_ly = ('52W_LY_num', 'sum')
        ).reset_index() # Add reset_index to bring 'Product Name' back as a column for direct use
        
        product_sales['abs_change_ty_minus_ly'] = product_sales['total_ty'] - product_sales['total_ly']
        
        product_sales = product_sales.sort_values(by='abs_change_ty_minus_ly', ascending=False)
        
        # Create dictionaries with Product Name as key and abs_change as value
        top_drivers_dict = dict(zip(product_sales.head(5)['Product Name'], product_sales.head(5)['abs_change_ty_minus_ly']))
        bottom_drivers_df = product_sales.tail(5).sort_values(by='abs_change_ty_minus_ly', ascending=True)
        bottom_drivers_dict = dict(zip(bottom_drivers_df['Product Name'], bottom_drivers_df['abs_change_ty_minus_ly']))

        summary["absolute_change_drivers_52w"] = {
            "metric_used": primary_metric_to_sum,
            "calculation_period": "52W",
            "top_5_positive_drivers": top_drivers_dict,
            "top_5_negative_drivers": bottom_drivers_dict
        }
    
    # Add product/category analysis if we have YoY results
    if yoy_results and 'Product Name' in data.columns:
        # Use 52W data for product ranking 
        if '52W_TY' in data.columns and '52W_LY' in data.columns:
            product_performance = {}
            for product in data['Product Name'].unique():
                product_data = data[data['Product Name'] == product]
                ty_sum = pd.to_numeric(product_data['52W_TY'], errors='coerce').fillna(0).sum()
                ly_sum = pd.to_numeric(product_data['52W_LY'], errors='coerce').fillna(0).sum()
                
                if ly_sum != 0:
                    product_yoy = ((ty_sum - ly_sum) / ly_sum) * 100
                else:
                    product_yoy = 0
                
                product_performance[product] = product_yoy
            
            # Sort products by performance
            sorted_products = sorted(product_performance.items(), key=lambda x: x[1], reverse=True)
            
            summary["product_performance"] = {
                "metric_used": "52W YoY Growth",
                "top_10_performers": dict(sorted_products[:10]),
                "bottom_10_performers": dict(sorted_products[-10:])
            }
    
    # Add category analysis
    if yoy_results and 'Category' in data.columns:
        category_performance = {}
        for category in data['Category'].unique():
            if pd.notna(category):
                category_data = data[data['Category'] == category]
                if '52W_TY' in data.columns and '52W_LY' in data.columns:
                    ty_sum = pd.to_numeric(category_data['52W_TY'], errors='coerce').fillna(0).sum()
                    ly_sum = pd.to_numeric(category_data['52W_LY'], errors='coerce').fillna(0).sum()
                    
                    if ly_sum != 0:
                        cat_yoy = ((ty_sum - ly_sum) / ly_sum) * 100
                    else:
                        cat_yoy = 0
                    
                    category_performance[category] = {
                        "yoy_growth": cat_yoy,
                        "product_count": len(category_data)
                    }
        
        summary["category_analysis"] = {
            "metric_used": "52W YoY Growth", 
            "categories": category_performance
        }
    
    # Store the summary in session state for caching
    st.session_state.yoy_summary = summary
    
    return summary


def handle_chatbot_question(question, data_summary, api_key):
    """Handle a chatbot question using the processed retail data - COPIED FROM WORKING EXAMPLE"""
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar to use the chatbot.")
        return
    
    # Add user message
    st.session_state.retail_chatbot_messages.append({"role": "user", "content": question})
    
    # Generate response
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Convert data to JSON string
        data_json = json.dumps(data_summary, indent=2, default=str) if data_summary else "No data available"
        
        system_prompt = """You are a retail data analyst expert specializing in YoY performance analysis.

You have access to processed retail data in JSON format with CALCULATED YoY metrics. Answer questions based ONLY on this data.

**Available YoY Calculations (in the 'yoy_calculations' section):**
For each time period (e.g., '52W_YoY', '12W_YoY', '4W_YoY'), the following are available:
1.  `overall_growth_primary_metric`: This is the headline YoY growth for 'Sales Value (£)'. It is calculated by SUMMING all 'Sales Value (£)' for TY, SUMMING all 'Sales Value (£)' for LY, and then performing the YoY calculation: `(Total_Sales_TY - Total_Sales_LY) / Total_Sales_LY * 100`. The raw `total_primary_metric_ty` and `total_primary_metric_ly` sums are also included.
2.  `primary_metric_for_overall`: Confirms that 'Sales Value (£)' was used for the `overall_growth_primary_metric`.
3.  `average_of_row_wise_growth_all_numeric`: This is the average of YoY% calculated for *ALL individual rows* where both TY and LY values for that specific period were numeric. Formula per row: `(TY_row - LY_row) / LY_row * 100` (LY=0 results in 0% YoY for that row).
4.  `std_dev_of_row_wise_growth_all_numeric`: Standard deviation of the `average_of_row_wise_growth_all_numeric` values, indicating volatility.
5.  `average_from_existing_column`: If your uploaded data had a pre-calculated YoY column for this period (e.g., '52 w/e change'), this is the average of that column (multiplied by 100 if it was a decimal). Its name is in `existing_column_name_used`. This can be used for comparison.
6.  Other stats related to `average_of_row_wise_growth_all_numeric`: `min_row_wise_growth_all_numeric`, `max_row_wise_growth_all_numeric`, `positive_rows_count_all_numeric`, `negative_rows_count_all_numeric`, `total_numeric_rows_for_avg`.

**Other Key Data Sections:**
*   `absolute_change_drivers_52w`: Identifies the top 5 products that contributed the most positive and most negative *absolute change* to 'Sales Value (£)' over the 52-week period. This shows which products had the biggest impact in terms of currency value change.
*   `product_performance`: Top/bottom performers based on overall 52W YoY growth percentage per product (calculated by summing TY/LY for all rows of a product, then doing YoY).
*   `category_analysis`: Category performance, similarly based on overall 52W YoY growth percentage per category.
*   `dataset_info`: Basic dataset metadata.

**How to Answer YoY Questions:**
*   **If asked for "overall YoY growth" or a general YoY trend without specifying a period:** 
    *   Provide the `overall_growth_primary_metric` (for 'Sales Value (£)') for ALL available time periods (52W, 12W, 4W). State the TY and LY totals if relevant.
    *   Then, provide the `average_of_row_wise_growth_all_numeric` for ALL available periods, perhaps mentioning its standard deviation for context on volatility.
    *   Also, if `average_from_existing_column` is available for any period, mention it and its source column for comparison.
*   **If asked for YoY growth for a SPECIFIC period (e.g., "What's the 12W YoY?"):**
    *   Provide `overall_growth_primary_metric`, `average_of_row_wise_growth_all_numeric` (and its std dev), and `average_from_existing_column` (if available) for that specific period.
*   **If asked about "biggest drivers" or "largest impact":** Refer to `absolute_change_drivers_52w` for changes in currency value, and `product_performance` for percentage-based top/bottom movers.

Always be precise with numbers and specify the period and calculation type when discussing YoY. 
Use emojis for readability (📈 for growth, 📉 for decline, 🏆 for top performers, etc.)."""
        
        user_prompt = f"""Based on the retail data below, please answer this question: {question}

Retail Data Analysis:
{data_json}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Add assistant response
        st.session_state.retail_chatbot_messages.append({"role": "assistant", "content": answer})
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        st.session_state.retail_chatbot_messages.append({"role": "assistant", "content": error_message})


def create_structured_export_data():
    """Create structured data for Excel export with YoY calculations, SKU rankings, and category analysis."""
    if 'data' not in st.session_state or st.session_state.data is None:
        return None
    
    data = st.session_state.data.copy()
    
    # Get the data summary with all calculations
    data_summary = get_data_summary()
    if not data_summary:
        return None
    
    export_data = {}
    
    # 1. MAIN DATA SHEET - Enhanced with calculated YoY metrics
    main_data = data.copy()
    
    # Add calculated YoY metrics for each period
    required_cols = ['Metric', '52W_TY', '52W_LY', '12W_TY', '12W_LY', '4W_TY', '4W_LY']
    if all(col in data.columns for col in required_cols):
        
        # Calculate YoY for each period
        for period_label, ty_col, ly_col in [("52W", "52W_TY", "52W_LY"), 
                                             ("12W", "12W_TY", "12W_LY"), 
                                             ("4W", "4W_TY", "4W_LY")]:
            
            # Convert to numeric
            ty_numeric = pd.to_numeric(main_data[ty_col], errors='coerce').fillna(0)
            ly_numeric = pd.to_numeric(main_data[ly_col], errors='coerce').fillna(0)
            
            # Calculate absolute change
            main_data[f'{period_label}_YoY_Abs_Change'] = ty_numeric - ly_numeric
            
            # Calculate percentage change
            main_data[f'{period_label}_YoY_Pct_Change'] = np.where(
                ly_numeric != 0,
                ((ty_numeric - ly_numeric) / ly_numeric) * 100,
                np.nan
            )
            
            # Add growth categories
            yoy_pct = main_data[f'{period_label}_YoY_Pct_Change']
            main_data[f'{period_label}_Growth_Category'] = np.where(
                yoy_pct.isna(), 'Unknown',
                np.where(yoy_pct >= 15, 'High Growth (15%+)',
                np.where(yoy_pct >= 5, 'Moderate Growth (5-15%)',
                np.where(yoy_pct >= -5, 'Stable (-5% to 5%)',
                np.where(yoy_pct >= -15, 'Moderate Decline (-5% to -15%)',
                'High Decline (-15%+)'))))
            )
    
    export_data['Main_Data_Enhanced'] = main_data
    
    # 2. YOY SUMMARY SHEET
    yoy_summary_rows = []
    if 'yoy_calculations' in data_summary:
        for period, calc_data in data_summary['yoy_calculations'].items():
            row = {
                'Period': period,
                'Calculation_Type': calc_data.get('calculation_type', ''),
                'Primary_Metric': calc_data.get('primary_metric_for_overall', ''),
                'Total_TY': calc_data.get('total_primary_metric_ty', 0),
                'Total_LY': calc_data.get('total_primary_metric_ly', 0),
                'Overall_Growth_Primary_Metric_Pct': calc_data.get('overall_growth_primary_metric', 0),
                'Avg_Row_Wise_Growth_Pct': calc_data.get('average_of_row_wise_growth_all_numeric', 0),
                'StdDev_Row_Wise_Growth': calc_data.get('std_dev_of_row_wise_growth_all_numeric', 0),
                'Min_Row_Growth_Pct': calc_data.get('min_row_wise_growth_all_numeric', 0),
                'Max_Row_Growth_Pct': calc_data.get('max_row_wise_growth_all_numeric', 0),
                'Positive_Rows_Count': calc_data.get('positive_rows_count_all_numeric', 0),
                'Negative_Rows_Count': calc_data.get('negative_rows_count_all_numeric', 0),
                'Total_Numeric_Rows': calc_data.get('total_numeric_rows_for_avg', 0),
                'Existing_Column_Avg': calc_data.get('average_from_existing_column', ''),
                'Existing_Column_Name': calc_data.get('existing_column_name_used', '')
            }
            yoy_summary_rows.append(row)
    
    export_data['YoY_Summary'] = pd.DataFrame(yoy_summary_rows)
    
    # 3. PRODUCT RANKINGS SHEET
    if 'Product Name' in data.columns and 'absolute_change_drivers_52w' in data_summary:
        product_rankings = []
        
        # Top positive drivers
        top_drivers = data_summary['absolute_change_drivers_52w'].get('top_5_positive_drivers', {})
        for rank, (product, change) in enumerate(top_drivers.items(), 1):
            product_rankings.append({
                'Rank': rank,
                'Product_Name': product,
                'Absolute_Change_52W': change,
                'Change_Type': 'Positive Driver',
                'Period': '52W'
            })
        
        # Top negative drivers
        bottom_drivers = data_summary['absolute_change_drivers_52w'].get('top_5_negative_drivers', {})
        for rank, (product, change) in enumerate(bottom_drivers.items(), 1):
            product_rankings.append({
                'Rank': rank,
                'Product_Name': product,
                'Absolute_Change_52W': change,
                'Change_Type': 'Negative Driver',
                'Period': '52W'
            })
        
        # Product performance rankings
        if 'product_performance' in data_summary:
            top_performers = data_summary['product_performance'].get('top_10_performers', {})
            for rank, (product, yoy_pct) in enumerate(top_performers.items(), 1):
                product_rankings.append({
                    'Rank': rank,
                    'Product_Name': product,
                    'YoY_Growth_Pct_52W': yoy_pct,
                    'Change_Type': 'Top Performer',
                    'Period': '52W'
                })
            
            bottom_performers = data_summary['product_performance'].get('bottom_10_performers', {})
            for rank, (product, yoy_pct) in enumerate(bottom_performers.items(), 1):
                product_rankings.append({
                    'Rank': rank,
                    'Product_Name': product,
                    'YoY_Growth_Pct_52W': yoy_pct,
                    'Change_Type': 'Bottom Performer',
                    'Period': '52W'
                })
        
        export_data['Product_Rankings'] = pd.DataFrame(product_rankings)
    
    # 4. CATEGORY ANALYSIS SHEET
    if 'category_analysis' in data_summary and data_summary['category_analysis'].get('categories'):
        category_rows = []
        categories = data_summary['category_analysis']['categories']
        
        for category, cat_data in categories.items():
            category_rows.append({
                'Category': category,
                'YoY_Growth_Pct_52W': cat_data.get('yoy_growth', 0),
                'Product_Count': cat_data.get('product_count', 0),
                'Metric_Used': data_summary['category_analysis'].get('metric_used', '52W YoY Growth')
            })
        
        # Sort by YoY growth
        category_df = pd.DataFrame(category_rows)
        category_df = category_df.sort_values('YoY_Growth_Pct_52W', ascending=False)
        category_df['Rank'] = range(1, len(category_df) + 1)
        
        # Add performance categories
        category_df['Performance_Category'] = np.where(
            category_df['YoY_Growth_Pct_52W'] >= 10, 'High Growth',
            np.where(category_df['YoY_Growth_Pct_52W'] >= 0, 'Growth',
            np.where(category_df['YoY_Growth_Pct_52W'] >= -10, 'Moderate Decline',
            'High Decline'))
        )
        
        export_data['Category_Analysis'] = category_df
    
    # 5. DETAILED PRODUCT DATA (separated by category if available)
    if 'Category' in data.columns:
        for category in data['Category'].unique():
            if pd.notna(category):
                category_data = data[data['Category'] == category].copy()
                
                # Add YoY calculations for this category subset
                if all(col in category_data.columns for col in required_cols):
                    for period_label, ty_col, ly_col in [("52W", "52W_TY", "52W_LY"), 
                                                         ("12W", "12W_TY", "12W_LY"), 
                                                         ("4W", "4W_TY", "4W_LY")]:
                        ty_numeric = pd.to_numeric(category_data[ty_col], errors='coerce').fillna(0)
                        ly_numeric = pd.to_numeric(category_data[ly_col], errors='coerce').fillna(0)
                        
                        category_data[f'{period_label}_YoY_Pct'] = np.where(
                            ly_numeric != 0,
                            ((ty_numeric - ly_numeric) / ly_numeric) * 100,
                            np.nan
                        )
                
                # Clean category name for sheet name (Excel sheet names have restrictions)
                clean_category_name = str(category).replace('/', '_').replace('\\', '_')[:31]
                export_data[f'Category_{clean_category_name}'] = category_data
    
    # 6. CALCULATION METADATA SHEET
    metadata_rows = [
        {'Field': 'Export_Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        {'Field': 'Total_Records', 'Value': len(data)},
        {'Field': 'Total_Columns', 'Value': len(data.columns)},
        {'Field': 'Primary_Metric_Used', 'Value': 'Sales Value (£)'},
        {'Field': 'YoY_Calculation_Method', 'Value': '(TY - LY) / LY * 100'},
        {'Field': 'Growth_Categories', 'Value': 'High Growth (15%+), Moderate Growth (5-15%), Stable (-5% to 5%), Moderate Decline (-5% to -15%), High Decline (-15%+)'},
        {'Field': 'Periods_Analyzed', 'Value': '52W, 12W, 4W'},
        {'Field': 'Data_Source_Columns', 'Value': ', '.join(data.columns.tolist())}
    ]
    
    export_data['Calculation_Metadata'] = pd.DataFrame(metadata_rows)
    
    return export_data


def export_to_excel(export_data):
    """Export the structured data to Excel format."""
    if not export_data:
        return None
    
    # Create Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in export_data.items():
            # Ensure sheet name is valid for Excel
            clean_sheet_name = sheet_name[:31]  # Excel limit
            df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[clean_sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output


def main():
    """Main Streamlit application - USING WORKING PATTERN"""
    st.set_page_config(
        page_title="Retail Data Analyst Chatbot",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Retail Data Analyst Chatbot")
    st.markdown("Upload your retail data and ask questions about YoY performance, trends, and insights!")
    
    # Initialize session state - COPIED FROM WORKING EXAMPLE
    if 'retail_chatbot_messages' not in st.session_state:
        st.session_state.retail_chatbot_messages = [
            {"role": "assistant", "content": "👋 Hi! I'm your retail data analyst. Upload your Excel or CSV file and I'll help you analyze YoY performance, identify trends, and uncover insights. What would you like to know?"}
        ]
    
    # Sidebar for file upload and API key
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input - COPIED FROM WORKING EXAMPLE
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable chatbot responses"
        )
        
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose your retail data file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload Excel or CSV files with retail data"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Analyze Data", type="primary"):
                success, message = load_and_process_data_directly(uploaded_file)
                if success:
                    st.success(message)
                    # Corrected boolean check for DataFrame
                    analyzed_records = len(st.session_state.data) if 'data' in st.session_state and st.session_state.data is not None else 'N/A'
                    analyzed_columns = len(st.session_state.data.columns) if 'data' in st.session_state and st.session_state.data is not None else 'N/A'
                    
                    st.session_state.retail_chatbot_messages.append({
                        "role": "assistant", 
                        "content": f"🎉 **Data Loaded & Processed!**\n\n📊 Analyzed {analyzed_records:,} records with {analyzed_columns} columns.\n\nReady for questions."
                    })
                    st.rerun() # This is important to refresh the app state
                else:
                    st.error(message)
                    # Ensure data is None if loading failed
                    st.session_state.data = None
                    if 'yoy_summary' in st.session_state:
                        del st.session_state.yoy_summary
        
        # Show data info if available
        if 'data' in st.session_state and st.session_state.data is not None:
            st.header("📊 Dataset Info")
            data = st.session_state.data
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Columns", len(data.columns))
            
            # Calculate and show YoY metrics
            data_summary = get_data_summary()
            if data_summary and 'yoy_calculations' in data_summary:
                yoy_count = len(data_summary['yoy_calculations'])
                st.metric("YoY Calculations", yoy_count)
                
                # Show brief summary of calculated metrics
                if yoy_count > 0:
                    st.write("**Calculated YoY:**")
                    for calc_name, calc_info in list(data_summary['yoy_calculations'].items())[:3]:  # Show max 3
                        if 'average_growth' in calc_info:
                            avg_growth = calc_info['average_growth']
                            st.write(f"• {calc_name}: {avg_growth:.1f}%")
            else:
                st.metric("YoY Calculations", 0)
            
            # Export functionality
            st.header("📤 Export Data")
            st.write("Export your processed data with YoY calculations, SKU rankings, and category analysis.")
            
            if st.button("🔄 Generate Structured Export", type="secondary"):
                with st.spinner("Creating structured export data..."):
                    export_data = create_structured_export_data()
                    
                    if export_data:
                        excel_file = export_to_excel(export_data)
                        
                        if excel_file:
                            # Generate filename with timestamp
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"retail_analysis_export_{timestamp}.xlsx"
                            
                            st.success(f"✅ Export ready! Contains {len(export_data)} sheets with comprehensive analysis.")
                            
                            # Show what's included in the export
                            st.write("**Export includes:**")
                            sheet_descriptions = {
                                'Main_Data_Enhanced': 'Original data + calculated YoY metrics',
                                'YoY_Summary': 'Summary of all YoY calculations',
                                'Product_Rankings': 'Top/bottom performers and drivers',
                                'Category_Analysis': 'Category performance rankings',
                                'Calculation_Metadata': 'Export details and methodology'
                            }
                            
                            for sheet_name in export_data.keys():
                                if sheet_name in sheet_descriptions:
                                    st.write(f"• **{sheet_name}**: {sheet_descriptions[sheet_name]}")
                                elif sheet_name.startswith('Category_'):
                                    st.write(f"• **{sheet_name}**: Detailed data for this category")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Excel File",
                                data=excel_file,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary"
                            )
                        else:
                            st.error("❌ Failed to create Excel file.")
                    else:
                        st.error("❌ No data available for export. Please load and analyze data first.")
        
        # Example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What's the overall YoY growth?",
            "Show me the top performing products",
            "Which products are declining?",
            "How are different categories performing?",
            "What are the key trends in the data?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                # Set a session state variable to trigger the question
                st.session_state.pending_question = question
                st.rerun()
    
    # Main chat interface - COPIED FROM WORKING EXAMPLE
    st.header("💬 Chat with your data")
    
    # Handle pending question from example buttons
    if 'pending_question' in st.session_state:
        pending = st.session_state.pending_question
        del st.session_state.pending_question
        
        if api_key:
            data_summary = get_data_summary()
            handle_chatbot_question(pending, data_summary, api_key)
        else:
            error_msg = "❌ Please enter your OpenAI API Key in the sidebar to use the chatbot."
            st.session_state.retail_chatbot_messages.append({"role": "user", "content": pending})
            st.session_state.retail_chatbot_messages.append({"role": "assistant", "content": error_msg})
    
    # Display chat messages from history
    for message in st.session_state.retail_chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input - USING EXACT WORKING PATTERN
    if prompt := st.chat_input("Ask me about your retail data..."):
        if api_key:
            data_summary = get_data_summary()
            handle_chatbot_question(prompt, data_summary, api_key)
        else:
            error_msg = "❌ Please enter your OpenAI API Key in the sidebar to use the chatbot."
            st.session_state.retail_chatbot_messages.append({"role": "user", "content": prompt})
            st.session_state.retail_chatbot_messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main() 
