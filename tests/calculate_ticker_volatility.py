import pandas as pd
import os
from nscan.config import RETURNS_DATA_DIR

def calculate_ticker_volatility(data_dir: str, start_year: int, end_year: int) -> pd.Series:
    """
    Calculate standard deviation of returns for each ticker across multiple years.
    
    Args:
        data_dir: Directory containing the yearly returns CSV files
        start_year: First year to include
        end_year: Last year to include (inclusive)
    
    Returns:
        pd.Series: Standard deviations indexed by ticker
    """
    # List to store DataFrames for each year
    yearly_dfs = []
    
    # Load and concatenate all yearly data
    for year in range(start_year, end_year + 1):
        file_path = data_dir / f"{year}_returns.csv"
        if not file_path.exists():
            print(f"Warning: File not found for year {year}")
            continue
            
        df = pd.read_csv(file_path)
        yearly_dfs.append(df)
    
    # Concatenate all years
    combined_df = pd.concat(yearly_dfs, ignore_index=True)
    
    # Pivot the data using Ticker instead of PERMNO
    pivoted_df = combined_df.pivot(
        index='DlyCalDt',
        columns='PERMNO',
        values='DlyRet'
    ).sort_index(axis=1)
    
    # Calculate standard deviation for each ticker, ignoring NA values
    ticker_volatility = pivoted_df.std(skipna=True)

    # Sort from low to high and add position
    sorted_volatility = ticker_volatility.sort_values(ascending=True)
    ranked_volatility = pd.DataFrame({
        'Volatility': sorted_volatility,
        'Position': range(1, len(sorted_volatility) + 1)
    })
    
    return ranked_volatility

if __name__ == "__main__":
    # Example usage
    data_dir = RETURNS_DATA_DIR
    start_year = 2006
    end_year = 2021
    
    volatility = calculate_ticker_volatility(data_dir, start_year, end_year)
    
    # Print results sorted by volatility (highest to lowest)
    print("\nnTicker Volatilities (sorted low to high):")
    print(volatility)
    
    # Optionally save to CSV
    volatility.to_csv(data_dir / 'ticker_volatilities.csv')