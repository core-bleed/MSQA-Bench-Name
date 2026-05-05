import os
import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode
import logging
import json
from openai import OpenAI, APIError

# --- Configuration ---
import argparse

_parser = argparse.ArgumentParser(description="Batch LLM-assisted PDF text cleaner.")
_parser.add_argument("--input-dir", default="data/input",
                     help="Directory of PDFs (default: data/input)")
_parser.add_argument("--output-dir", default="data/output_llm",
                     help="Output directory (default: data/output_llm)")
_parser.add_argument("--log-file", default="logs/processing.log",
                     help="Log file (default: logs/processing.log)")
_parser.add_argument("--record", default="processed_files.json",
                     help="Resume record (default: processed_files.json)")
_args = _parser.parse_args()

input_directory = _args.input_dir
output_directory = _args.output_dir
log_file = _args.log_file
processed_files_record = _args.record
os.makedirs(output_directory, exist_ok=True)
os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
key = os.environ.get("OPENAI_API_KEY", "")

# --- Initialize Logging ---
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
model_name = "o3-mini"  # Specify the model name here
# --- Initialize OpenAI Client for Ollama ---
try:
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    client = OpenAI(

        api_key=key,
    )
    logging.info(
        f"OpenAI client initialized to target Ollama at http://localhost:11434/v1"
    )
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    exit()

# --- Load Processed Files Record ---
try:
    if os.path.exists(processed_files_record):
        with open(processed_files_record, "r", encoding="utf-8") as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()
except Exception as e:
    logging.error(f"Error loading processed files record: {e}")
    processed_files = set()


# --- PDF Processing ---
def process_pdf(pdf_path, output_file):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {e}")
        return

    all_blocks = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        for block in blocks:
            if block[6] == 0:  # block_type 0 is text
                text = block[4].strip().replace("\r", "\n").replace("\n\n", "\n")
                if text:
                    all_blocks.append((page_num, block[5], text))

    # Convert to DataFrame
    df = pd.DataFrame(all_blocks, columns=["page", "block_no", "text"])
    df = df.sort_values(by=["page", "block_no"]).reset_index(drop=True)
    df["text"] = df["text"].apply(unidecode)

    all_cleaned_text = []
    for page_num, group in df.groupby("page"):
        page_text = "\n".join(group["text"])

        if not page_text.strip():
            logging.info(f"  Skipping Page {page_num} (no text content).")
            continue

        # Prepare the Prompt for Ollama
        prompt = f"""        
        Extract and clean the raw text content from this PDF page following these strict rules:
        1. Remove ALL:
        - References/citations (e.g., [1], (Smith 2020))
        - Figure/table mentions (e.g., "Figure 1:", "Table 2 shows...")
        - Captions, footnotes, or marginalia
        - Page numbers/headers/footers
        - Stray characters, symbols, or OCR artifacts

        2. Preserve ONLY:
        - Main body text paragraphs
        - Natural paragraph breaks (single blank line between paragraphs)
        - Corrected OCR errors ONLY when absolutely certain (e.g., "teh" → "the")

        3. Formatting:
        - Single spaces between words
        - Single line breaks between paragraphs
        - No leading/trailing whitespace

        4. Absolutely DO NOT:
        - Add any commentary, explanations, or metadata
        - Summarize or rephrase content
        - Include any non-text elements
        - Create placeholder text like "[...]"

        Return ONLY the cleaned text content with no additional text from you.
        Text to clean:
        --- START TEXT ---
        {page_text}
        --- END TEXT ---

        Cleaned text:"""
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that cleans text according to specific rules and outputs the result *only* .You do not provide explanations or thoughts.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                full_response_content = response.choices[0].message.content
                text = (
                    full_response_content.replace("\\n", "\n")
                    .replace("\n", " ")
                    .strip()
                )
                all_cleaned_text.append(text)
                logging.info(f"    Page {page_num} processed successfully.")
            else:
                logging.warning(
                    f"    Warning: Received empty or unexpected response for Page {page_num}."
                )
                all_cleaned_text.append(page_text)

        except Exception as e:
            logging.error(
                f"    An unexpected error occurred during API response processing for Page {page_num}: {e}"
            )
            all_cleaned_text.append(page_text)

    final_cleaned_text = "\n\n".join(all_cleaned_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_cleaned_text)

    doc.close()
    logging.info(f"\nProcessing complete. Cleaned text saved to '{output_file}'.")


# --- Main Logic ---
for filename in os.listdir(input_directory):
    if filename.endswith(".pdf") and filename not in processed_files:
        pdf_path = os.path.join(input_directory, filename)
        output_file = os.path.join(
            output_directory, f"{os.path.splitext(filename)[0]}.txt"
        )
        
        process_pdf(pdf_path, output_file)

        # Mark the file as processed
        processed_files.add(filename)

        # Update processed files record
        try:
            with open(processed_files_record, "w", encoding="utf-8") as f:
                json.dump(list(processed_files), f)
        except Exception as e:
            logging.error(f"Error updating processed files record: {e}")

logging.info("All PDFs processed.")
