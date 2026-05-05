import os
import requests
import xml.etree.ElementTree as ET
import logging

# Configure logging to write to 'process.log'
# Change 'process.log' to any other filename/path as needed.
logging.basicConfig(
    filename="process.log",
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1


def extract_xml_using_grobid(pdf_path):
    """Extract XML from a PDF using Grobid."""
    url = "http://localhost:8070/api/processFulltextDocument"  # URL for local Grobid server

    with open(pdf_path, "rb") as pdf_file:
        files = {"input": pdf_file}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.text  # Return the full XML extracted from the PDF
    else:
        raise Exception(
            f"Failed to extract XML from PDF '{pdf_path}'. "
            f"Status code: {response.status_code} - {response.text}"
        )


def extract_text_from_grobid_xml(xml_content):
    """Extract text from Grobid XML output."""
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    extracted_text = ""
    in_conclusions = False
    processed_divs = set()

    # Loop through elements that contain textual content (abstract, body, div, p)
    for elem in root.iter():
        if elem.tag == "{http://www.tei-c.org/ns/1.0}abstract":
            text = elem.text.strip() if elem.text else ""
            if text:
                extracted_text += "Abstract: " + text + "\n\n"

        elif elem.tag == "{http://www.tei-c.org/ns/1.0}body":
            for child in elem.iter("{http://www.tei-c.org/ns/1.0}p"):
                text = child.text.strip() if child.text else ""
                if text:
                    extracted_text += text + "\n"

        elif elem.tag == "{http://www.tei-c.org/ns/1.0}div":
            div_id = id(elem)
            if div_id in processed_divs:
                continue  # Skip if this <div> has already been processed

            for child in elem.iter("{http://www.tei-c.org/ns/1.0}head"):
                if child.text and "Conclusions" in child.text:
                    in_conclusions = True
                    extracted_text += "\nConclusions:\n"

            if in_conclusions:
                for child in elem.iter("{http://www.tei-c.org/ns/1.0}p"):
                    text = child.text.strip() if child.text else ""
                    if text:
                        extracted_text += text + "\n"
                processed_divs.add(div_id)  # Mark this <div> as processed

        elif elem.tag in (
            "{http://www.tei-c.org/ns/1.0}figure",
            "{http://www.tei-c.org/ns/1.0}table",
        ):
            # Skip figures and tables
            continue

    return extracted_text


def save_text_to_file(text, output_file):
    """Save the extracted text into a .txt file."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)


def process_pdfs_in_directory(input_dir, output_dir):
    """
    Process all PDFs in a directory and save the extracted text.
    Skips any PDF if its corresponding TXT file already exists.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    total = len(pdf_files)
    processed_count = 0

    logger.info(f"Found {total} PDF files in '{input_dir}'. Starting processing...")

    for index, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(input_dir, pdf_file)
        txt_filename = f"{os.path.splitext(pdf_file)[0]}.txt"
        output_file = os.path.join(output_dir, txt_filename)

        # Skip if output .txt already exists
        if os.path.exists(output_file):
            logger.info(f"[{index}/{total}] Skipping '{pdf_file}' (already processed).")
            continue

        logger.info(f"[{index}/{total}] Processing '{pdf_file}'...")

        try:
            # Step 1: Extract XML from PDF using Grobid
            extracted_xml = extract_xml_using_grobid(pdf_path)

            # Step 2: Extract text from the Grobid XML
            extracted_text = extract_text_from_grobid_xml(extracted_xml)

            # Step 3: Save the extracted text into a .txt file
            save_text_to_file(extracted_text, output_file)

            processed_count += 1
            logger.info(
                f"      Saved extracted text for '{pdf_file}' to '{output_file}'"
            )

        except Exception as e:
            logger.error(f"      Error processing '{pdf_file}': {e}")

    logger.info(
        f"Done. Processed {processed_count} new files out of {total} PDF files."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GROBID full-text extraction over a PDF directory (batch)."
    )
    parser.add_argument("--input-dir", default="data/input",
                        help="Directory of PDFs (default: data/input)")
    parser.add_argument("--output-dir", default="data/grobid",
                        help="Output directory (default: data/grobid)")
    args = parser.parse_args()

    process_pdfs_in_directory(args.input_dir, args.output_dir)
