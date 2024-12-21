import pandas as pd
import re
from datetime import datetime

# Load the CSV files
df_v2 = pd.read_csv('FNSPID_v2.csv')
df_v1 = pd.read_csv('FNSPID_v1.csv')
df_top6000 = pd.read_csv('Financial_news_top6000.csv')

# Function to extract date from the first 5 words of the article
def extract_date(text):
    if isinstance(text, str):
        match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}', ' '.join(text.split()[:5]))
        return match.group() if match else None
    return None

# Function to convert extracted date to datetime object
def convert_date(date_str):
    try:
        return datetime.strptime(date_str + " 2023", "%b %d %Y") if date_str else None
    except ValueError:
        return None

# Clean and preprocess datasets
def preprocess_dataset(df, drop_columns):
    df = df.dropna(subset=drop_columns)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

df_v2 = preprocess_dataset(df_v2, ['Article','Url', 'Date', 'Article_title'])
df_v1 = preprocess_dataset(df_v1, ['Article', 'Url', 'Date', 'Article_title'])
df_top6000 = preprocess_dataset(df_top6000, ['Article', 'Url', 'Date', 'Article_title'])

# Function to update dates based on matches in another dataset and extracted dates from article text
def update_dates(source_df, target_df):
    for index, row in target_df.iterrows():
        match = source_df[
            (source_df['Article_title'] == row['Article_title']) |
            (source_df['Author'] == row['Author']) |
            (source_df['Url'] == row['Url'])
        ]
        if not match.empty and pd.notna(match.iloc[0]['Date']) and row['Date'] != match.iloc[0]['Date']:
            target_df.at[index, 'Date'] = match.iloc[0]['Date']
        
        # Extract and update dates from the article text if no date match is found
        if pd.isna(target_df.at[index, 'Date']):
            extracted_date = extract_date(row['Article'])
            converted_date = convert_date(extracted_date)
            if pd.notna(converted_date):
                target_df.at[index, 'Date'] = converted_date

# Update FNSPID_v2 dates based on FNSPID_v1 and Financial_news_top6000
update_dates(df_v1, df_v2)
update_dates(df_top6000, df_v2)

# Save the updated datasets
df_v2.to_csv('Updated_FNSPID_v2.csv', index=False)
df_v1.to_csv('Cleaned_FNSPID_v1.csv', index=False)
df_top6000.to_csv('Cleaned_Financial_news_top6000.csv', index=False)

print("FNSPID_v2 has been updated based on FNSPID_v1, Financial_news_top6000, and extracted dates from article text.")
