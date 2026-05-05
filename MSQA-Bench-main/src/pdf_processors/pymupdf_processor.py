import argparse

import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode

parser = argparse.ArgumentParser(description="Extract block-level text from a PDF using PyMuPDF.")
parser.add_argument("pdf", nargs="?", default="data/input/sample.pdf",
                    help="Path to a PDF file (default: data/input/sample.pdf)")
parser.add_argument("--out-txt", default="output.txt", help="Output text file")
parser.add_argument("--out-csv", default="output.csv", help="Output CSV file")
args = parser.parse_args()

pdf_path = args.pdf

# Open the PDF
doc = fitz.open(pdf_path)

# Initialize an empty list to hold all text blocks
all_blocks = []

# Loop through each page in the document
for page_num, page in enumerate(doc, start=1):
    # Get the text blocks with sort=True to handle multi-column layouts
    blocks = page.get_text("blocks", sort=True)
    for block in blocks:
        if block[6] == 0:  # Only include text blocks
            # Append page number and block details
            all_blocks.append((page_num,) + block)

# Define the columns for the DataFrame
columns = ['page', 'x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'block_type']

# Create the DataFrame
df = pd.DataFrame(all_blocks, columns=columns)

# Apply unidecode to the 'text' column to handle Unicode characters
df['text'] = df['text'].apply(unidecode)

# Reset the index
df = df.reset_index(drop=True)


with open(args.out_txt, 'w', encoding='utf-8') as file:
    for index, row in df.iterrows():
        file.write(f"Page: {row['page']}, Block: {row['block_no']}, Text: {row['text']}\n")

# Option 2: Save as a CSV file (recommended for structured data)
df.to_csv(args.out_csv, index=False, encoding='utf-8')

# Close the document
doc.close()

print(f"DataFrame has been saved to '{args.out_txt}' and '{args.out_csv}'.")