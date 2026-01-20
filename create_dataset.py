import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import os

PARQUET_FILE = "laion-coco_v3_filter.parquet"
OUTPUT_FILENAME = "prompts.csv"
TARGET_COUNT = 1_000_000

print(f"--- Loading from local parquet file: {PARQUET_FILE} ---")

if not os.path.exists(PARQUET_FILE):
    print(f"Error: File '{PARQUET_FILE}' not found!")
    print("Please check the path and try again.")
    exit(1)

try:
    print("Inspecting parquet file...")
    parquet_file = pq.ParquetFile(PARQUET_FILE)
    
    print(f"File metadata:")
    print(f"  Total rows: {parquet_file.metadata.num_rows:,}")
    print(f"  Columns: {parquet_file.schema.names}")
    print(f"  File size: {os.path.getsize(PARQUET_FILE) / (1024**3):.2f} GB")
    
    possible_text_cols = ["TEXT", "text", "caption", "prompt", "CAPTION", "PROMPT", "string"]
    text_column = None
    
    for col in possible_text_cols:
        if col in parquet_file.schema.names:
            text_column = col
            break
    
    if not text_column:
        print(f"Could not identify text column. Available columns: {parquet_file.schema.names}")
        text_column = input("Please enter the name of the text column from the list above: ").strip()
        if text_column not in parquet_file.schema.names:
            print(f"Column '{text_column}' not found in the file.")
            exit(1)
    
    print(f"Using text column: '{text_column}'")
    
    prompts = []
    total_rows = parquet_file.metadata.num_rows
    rows_to_process = min(TARGET_COUNT, total_rows)
    CHUNK_SIZE = 50000 
    
    print(f"Processing {rows_to_process:,} rows in chunks of {CHUNK_SIZE:,}...")
    
    with tqdm(total=rows_to_process) as pbar:
        for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=[text_column]):
            batch_df = batch.to_pandas()
            
            for text in batch_df[text_column]:
                if len(prompts) >= rows_to_process:
                    break
                    
                if text and isinstance(text, str) and len(text.strip()) > 5:
                    cleaned_text = text.replace("\n", " ").strip()
                    cleaned_text = ' '.join(cleaned_text.split())
                    prompts.append(cleaned_text)
                    pbar.update(1)
            
            if len(prompts) >= rows_to_process:
                break
                
            if len(prompts) % 100000 == 0 and len(prompts) > 0:
                print(f"  Progress: {len(prompts):,} prompts collected")
    
    print(f"\nCollected {len(prompts):,} valid prompts")
    
    print(f"Saving to {OUTPUT_FILENAME}...")
    output_df = pd.DataFrame({"text": prompts})
    output_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"Success! Saved {len(prompts):,} prompts to {OUTPUT_FILENAME}")
    if prompts:
        avg_length = sum(len(p) for p in prompts) / len(prompts)
        print(f"Average prompt length: {avg_length:.1f} characters")
        print(f"Sample prompts:")
        for i in range(min(3, len(prompts))):
            print(f"  {i+1}. {prompts[i][:100]}...")

except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure pyarrow is installed: pip install pyarrow")
    print("2. Check if the file is corrupted")
    print("3. Try with a smaller batch size if memory is an issue")