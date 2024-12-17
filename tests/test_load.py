from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

def test_pipeline():
    # 1. Load FNSPID dataset
    print("Loading FNSPID dataset...")
    try:
        dataset = load_dataset("csv", data_files="data/raw/FNSPID-date-corrected.csv", split="train")

        # non_empty_count = len(dataset.filter(lambda x: x['Article'] != "" and x['Article'] is not None))
        # print(f"Number of non-empty articles: {non_empty_count}")
        # print(f"Total number of articles: {len(dataset)}")
        # print(f"Percentage non-empty: {(non_empty_count/len(dataset))*100:.2f}%")

        # Chunked pandas approach for year counting
        year_counts = pd.Series(dtype=int)  # Initialize empty series
        chunk_size = 100000 # Adjust based on memory and performance

        print("Counting articles by year...")
        for i, chunk in enumerate(pd.read_csv("data/raw/FNSPID-date-corrected.csv", chunksize=chunk_size)):
            # Process each chunk
            chunk_counts = chunk['Date'].str[:4].value_counts()
            year_counts = year_counts.add(chunk_counts, fill_value=0)
            
            # Print progress
            rows_processed = (i + 1) * chunk_size
            print(f"Processed {rows_processed:,} rows...", end='\r')

        year_counts = year_counts.sort_index()
        print("\nDone!")
        print("\nYear counts:")
        print(year_counts)

        year_counts.plot(kind='bar')
        plt.title('Number of Articles by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nDataset sample:")
        print(dataset[0])  # Let's see what fields we have
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    # 2. Load FinText model
    print("\nLoading FinText model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")
        model = AutoModel.from_pretrained("FinText/FinText-Base-2007")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 3. Process an example
    print("\nProcessing example text...")
    example_text = dataset[0:5]['Article']  # Adjust field name if needed
    print(f"Example text: {example_text[1][:200]}...")  # Show first 200 chars
    
    # 4. Tokenize and get model output
    inputs = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    for i, text in enumerate(example_text):
        print(f"\n=== Article {i+1} ===")
        print("\nOriginal text length (chars):", len(text))
        print("Original first 150 chars:", text[:150])
        
        # Decode the tokenized text to see what actually gets processed
        decoded_text = tokenizer.decode(inputs['input_ids'][i])
        print("\nTokenized text length (chars):", len(decoded_text))
        print("Processed first 150 chars:", decoded_text[:150])
        
        # Show if truncation occurred
        if len(text) > len(decoded_text):
            print(f"\n*** Truncated! Lost {len(text) - len(decoded_text)} characters ***")
            print("\nLast 150 chars that made it in:", decoded_text[-150:])
            print("\nFirst 150 chars that were cut off:", text[len(decoded_text):len(decoded_text)+150])
            
    outputs = model(**inputs)
    
    # 5. Print info about shapes and sizes
    print("\nModel output shape:", outputs.last_hidden_state.shape)
    print("Number of dataset examples:", len(dataset))
    
    return dataset, model, tokenizer

if __name__ == "__main__":
    dataset, model, tokenizer = test_pipeline()