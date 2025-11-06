import os
import glob
import pandas as pd
from google import genai
from google.genai import types
import json 
from pydantic import BaseModel, Field
from enum import Enum 
import time  # <-- NEW: Import the time module

# --- Pydantic Schema Definition (Matching Original Column Names) ---

# Define the 6 core legal domains as an Enum 
class ApplicableLaw(str, Enum):
    FAMILY_LAW = "Family Law"
    PROPERTY_LAW = "Property Law"
    CONSUMER_LAW = "Consumer Law"
    CYBER_LAW = "Cyber Law"
    WOMEN_CHILD = "Women & Child"
    LABOUR_LAW = "Labour Law"

class ClassificationSchema(BaseModel):
    """Schema for classifying and summarizing a legal case judgment, 
    matching the requested output column names."""
    
    primary_classification: ApplicableLaw = Field(
        description="The primary, highest-level legal domain this case belongs to. Must be one of the six categories defined."
    )
    subclass: str = Field(
        description="The specific sub-category or precise legal matter (e.g., Divorce, Negligence, IT Act Section 66)."
    )
    case_summary_text: str = Field(
        description="A concise, factual summary of the case facts and the court's final ruling. Must be between 200 and 400 words. If the source text is insufficient, output 'SUMMARY_TOO_SHORT'."
    )
# --- End Schema Definition ---


# --- Configuration ---
INPUT_PDF_DIR = 'files'  
OUTPUT_FILE = 'legal_dataset.csv' 
MAX_FILES_TO_PROCESS = 250 # TARGET LIMIT SET TO 250 FILES
MODEL_NAME = "gemini-2.5-flash"
SLEEP_TIME_SECONDS = 5 # <-- NEW: Set a sleep duration to manage rate limits

# Use the Pydantic model to get the JSON schema for the API
GEMINI_SCHEMA = ClassificationSchema.model_json_schema()
CLASSIFICATION_LABELS = [e.value for e in ApplicableLaw]

# Original column names
OUTPUT_COLUMNS = ['file_name', 'primary_classification', 'subclass', 'case_summary_text', 'status']
# ---


def process_pdfs_in_batches():
    print("--- Starting Gemini Labeling Batch (Max 250 Files) ---")
    
    # 1. Setup Client and Check API Key
    try:
        if "GEMINI_API_KEY" not in os.environ:
            print("❌ ERROR: GEMINI_API_KEY environment variable is not set. Please set it.")
            return

        client = genai.Client()
    except Exception as e:
        print(f"❌ ERROR initializing Gemini Client: {e}")
        return

    # 2. Prepare File List and Resume Logic (Unchanged)
    pdf_files = glob.glob(os.path.join(INPUT_PDF_DIR, '*.pdf'))
    pdf_files.sort() 
    
    # --- STAGE 1: CHECK RESUMPTION & WRITE HEADER IF NECESSARY ---
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_df = existing_df.dropna(subset=['file_name'])
            processed_count = len(existing_df)
        except (pd.errors.EmptyDataError, Exception):
             processed_count = 0
             
    if processed_count == 0:
        header_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        header_df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)
        print(f"✅ Created new output file with header: {OUTPUT_FILE}")
    else:
        print(f"Resuming process. Found **{processed_count}** already processed entries in {OUTPUT_FILE}.")
    # --- END STAGE 1 ---


    # 3. Determine files for the CURRENT batch
    files_to_process_now = pdf_files[processed_count : MAX_FILES_TO_PROCESS]
    
    if not files_to_process_now:
        print(f"✅ Batch complete. Already processed {processed_count} files (Target: {MAX_FILES_TO_PROCESS}).")
        return

    docs_to_run_in_batch = len(files_to_process_now)
    print(f"Processing {docs_to_run_in_batch} new files, starting from file **#{processed_count + 1}**...")
    # NOTE: The estimated run time now includes the sleep delay.
    total_batch_time_min = (docs_to_run_in_batch * (10 + SLEEP_TIME_SECONDS)) / 60
    print(f"Expected run time for this batch (including {SLEEP_TIME_SECONDS}s delay): approx. **{total_batch_time_min:.1f}** minutes.")

    # 4. Processing Loop
    for i, file_path in enumerate(files_to_process_now):
        global_index = processed_count + i + 1
        
        print(f"[{global_index}/{MAX_FILES_TO_PROCESS}] Processing **{os.path.basename(file_path)}**...")

        row = {
            'file_name': os.path.basename(file_path),
            'primary_classification': 'PENDING',
            'subclass': 'PENDING',
            'case_summary_text': 'Processing...',
            'status': 'FAILED'
        }
        
        uploaded_file = None 

        try:
            # Upload PDF file to Gemini API (CRITICAL for large files)
            uploaded_file = client.files.upload(file=file_path)

            prompt = (
                f"Analyze the attached High Court judgment document. Your task is to perform Zero-Shot Classification and summarization. "
                f"Strictly adhere to the output JSON schema and the following rules: "
                f"1. PRIMARY_CLASSIFICATION: You MUST select the most relevant domain from: {', '.join(CLASSIFICATION_LABELS)}. "
                f"2. SUMMARY: The summary (case_summary_text) must be between 200 and 400 words. If the document is too short or lacks sufficient detail, set the summary text to 'SUMMARY_TOO_SHORT' (DO NOT create a short summary in this case)."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GEMINI_SCHEMA, 
                    temperature=0.0
                )
            )

            result = json.loads(response.text)
            
            # Update row with successful data
            primary_class = result.get('primary_classification', 'N/A')
            row.update({
                'primary_classification': primary_class,
                'subclass': result.get('subclass', 'N/A'),
                'case_summary_text': result.get('case_summary_text', 'N/A'),
                'status': 'SUCCESS'
            })
            print(f"   > Classified as: {primary_class} / Subclass: {result.get('subclass', 'N/A')}")
            
            # 🔑 CRITICAL FOR FREE TIER: Delete immediately to free up file storage quota!
            client.files.delete(name=uploaded_file.name)
            print("   > File successfully deleted from API service.")


        except Exception as e:
            # If an error occurred (API failure, JSON parsing, etc.)
            row.update({
                'primary_classification': 'ERROR',
                'subclass': 'ERROR',
                'case_summary_text': str(e),
                'status': 'FAILED'
            })
            print(f"   --> FAILED to process file {os.path.basename(file_path)}. Error: {e}")
            
            # Attempt to delete the file even on failure, if it was uploaded
            if uploaded_file:
                 try:
                    client.files.delete(name=uploaded_file.name)
                    print("   > File deleted from API service (after failure).")
                 except Exception as del_e:
                     print(f"   > Warning: Could not delete file {uploaded_file.name} due to error: {del_e}")
            
        # --- CRITICAL SAVE: SAVE TO CSV IMMEDIATELY ---
        new_row_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS) 
        new_row_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print(f"[PROGRESS SAVED] File {global_index} written to CSV.")
        # --- END CRITICAL SAVE ---

        # ⏳ RATE LIMIT CONTROL: Pause before processing the next file
        print(f"😴 Pausing for {SLEEP_TIME_SECONDS} seconds to respect API rate limits...")
        time.sleep(SLEEP_TIME_SECONDS)
            
    # 5. Final Summary
    final_df = pd.read_csv(OUTPUT_FILE)
    final_count = len(final_df.dropna(subset=['file_name']))
    print(f"\n--- Batch Run Finished ---")
    print(f"Total labeled entries saved in **{OUTPUT_FILE}**: {final_count}")


if __name__ == "__main__":
    process_pdfs_in_batches()