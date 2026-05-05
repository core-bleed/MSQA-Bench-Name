import argparse
import os

import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode
import json
from openai import OpenAI, APIError
import re

# --- Configuration ---
_parser = argparse.ArgumentParser(description="LLM-assisted PDF text cleaner (page by page).")
_parser.add_argument("pdf", nargs="?", default="data/input/sample.pdf",
                     help="Path to a PDF file (default: data/input/sample.pdf)")
_parser.add_argument("--output", default="cleaned_output.txt", help="Cleaned output file")
_parser.add_argument("--model", default="o3-mini", help="Chat model name (e.g. o3-mini, gpt-4o-mini)")
_args = _parser.parse_args()

pdf_path = _args.pdf
output_cleaned_file = _args.output
ollama_model = _args.model
key = os.environ.get("OPENAI_API_KEY", "")
if not key:
    raise RuntimeError("Set OPENAI_API_KEY before running this script.")


# --- Initialize OpenAI Client ---
try:
    client = OpenAI(
        api_key=key,
    )

    print(f"OpenAI client initialized to target Ollama at ")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

# --- PDF Parsing (Your existing code) ---
try:
    doc = fitz.open(pdf_path)
except Exception as e:
    print(f"Error opening PDF: {e}")
    exit()

all_blocks = []
for page_num, page in enumerate(doc, start=1):
    # Get text blocks, sorted for better reading order on the page
    blocks = page.get_text("blocks", sort=True)
    for block in blocks:
        # block format: (x0, y0, x1, y1, "text", block_no, block_type)
        if block[6] == 0:  # block_type 0 is text
            # Clean up text slightly - remove leading/trailing whitespace and replace multiple newlines
            text = block[4].strip().replace("\r", "\n").replace("\n\n", "\n")
            if text:  # Only add if there's actual text after stripping
                # Apply unidecode *here* if desired, or later
                # text = unidecode(text)
                all_blocks.append(
                    (page_num, block[5], text)
                )  # page_num, block_no, text

# Define the columns for the DataFrame
columns = ["page", "block_no", "text"]

# Create the DataFrame
df = pd.DataFrame(all_blocks, columns=columns)

# Ensure blocks are sorted correctly by page and then block number
df = df.sort_values(by=["page", "block_no"]).reset_index(drop=True)

# Optional: Apply unidecode to the entire column if preferred
df["text"] = df["text"].apply(unidecode)

# --- Chunking and Processing with Ollama ---
df.to_csv("output.csv", index=False, encoding="utf-8")

all_cleaned_text = []

print(f"Processing {df['page'].nunique()} pages...")

# Group by page number to process page by page
for page_num, group in df.groupby("page"):
    print(f"  Processing Page {page_num}...")

    # Combine text blocks for the current page
    page_text = "\n".join(group["text"])

    if not page_text.strip():
        print(f"    Skipping Page {page_num} (no text content).")
        continue

    # --- Prepare the Prompt for Ollama ---
    # Be very specific about the cleaning task.
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

    # --- Call Ollama API ---
    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            # temperature=0.1, # Optional: Lower temp for more deterministic cleaning
        )

        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            full_response_content = response.choices[0].message.content

            # Clean up the response text
            text = full_response_content.replace("\\n", "\n").replace("\n", " ").strip()

            all_cleaned_text.append(text)
            print(f"    Page {page_num} processed successfully.")

        else:
            # The API response structure itself was invalid or empty
            print(
                f"    Warning: Received empty or unexpected response structure from API for Page {page_num}."
            )
            print(
                f"    Using original text for Page {page_num} due to empty/invalid response."
            )
            all_cleaned_text.append(page_text)  # Fallback

    except Exception as e:  # General catch-all for other unexpected errors
        print(
            f"    An unexpected error occurred during API response processing for Page {page_num}: {e}"
        )
        print(f"    Using original text for Page {page_num} due to unexpected error.")
        all_cleaned_text.append(page_text)  # Fallback

# --- Combine and Save Final Output ---
final_cleaned_text = "\n\n".join(
    all_cleaned_text
)  # Join cleaned pages with double newline for separation

with open(output_cleaned_file, "w", encoding="utf-8") as f:
    f.write(final_cleaned_text)

# --- Close PDF ---
doc.close()

print(f"\nProcessing complete. Cleaned text saved to '{output_cleaned_file}'.")
# Optional: Save the intermediate DataFrame if needed
# df.to_csv('intermediate_extracted_blocks.csv', index=False, encoding='utf-8')
# print("Intermediate DataFrame saved to 'intermediate_extracted_blocks.csv'.")
