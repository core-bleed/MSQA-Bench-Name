import os
import requests
import xml.etree.ElementTree as ET

# docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1
# Define the namespace for the TEI format
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def extract_xml_using_grobid(pdf_path):
    """Extract XML from a PDF using Grobid"""
    url = 'http://localhost:8070/api/processFulltextDocument'  # URL for local Grobid server
    files = {'input': open(pdf_path, 'rb')}
    
    response = requests.post(url, files=files)

    if response.status_code == 200:
        return response.text  # Return the full XML extracted from the PDF
    else:
        raise Exception(f"Failed to extract XML from PDF. Status code: {response.status_code}")


def extract_text_from_grobid_xml(xml_content):
    """Extract text from Grobid XML output"""
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    extracted_text = ""
    in_conclusions = False
    processed_divs = set()

    # Loop through elements that contain textual content (e.g., abstract, body, div, p)
    for elem in root.iter():
        if elem.tag == '{http://www.tei-c.org/ns/1.0}abstract':
            text = elem.text.strip() if elem.text else ""
            if text:
                extracted_text += "Abstract: " + text + "\n\n"

        elif elem.tag == '{http://www.tei-c.org/ns/1.0}body':
            for child in elem.iter('{http://www.tei-c.org/ns/1.0}p'):
                text = child.text.strip() if child.text else ""
                if text:
                    extracted_text += text + "\n"

        elif elem.tag == '{http://www.tei-c.org/ns/1.0}div':
            div_id = id(elem)
            if div_id in processed_divs:
                continue  # Skip if this <div> has already been processed

            for child in elem.iter('{http://www.tei-c.org/ns/1.0}head'):
                if child.text and "Conclusions" in child.text:
                    in_conclusions = True
                    extracted_text += "\nConclusions:\n"

            if in_conclusions:
                for child in elem.iter('{http://www.tei-c.org/ns/1.0}p'):
                    text = child.text.strip() if child.text else ""
                    if text:
                        extracted_text += text + "\n"
                processed_divs.add(div_id)  # Mark this <div> as processed

        elif elem.tag == '{http://www.tei-c.org/ns/1.0}figure' or elem.tag == '{http://www.tei-c.org/ns/1.0}table':
            continue  # Skip figures and tables

    return extracted_text

def save_text_to_file(text, output_file):
    """Save the extracted text into a .txt file"""
    with open(output_file, 'w') as file:
        file.write(text)

def process_pdfs_in_directory(input_dir, output_dir):
    """Process all PDFs in a directory and save the extracted text"""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all PDF files in the input directory
    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"Processing {pdf_file}...")

            try:
                # Step 1: Extract XML from PDF using Grobid
                extracted_xml = extract_xml_using_grobid(pdf_path)

                # Step 2: Extract text from the Grobid XML
                extracted_text = extract_text_from_grobid_xml(extracted_xml)

                # Step 3: Save the extracted text into a .txt file
                output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.txt")
                save_text_to_file(extracted_text, output_file)

                print(f"Saved extracted text for {pdf_file} to {output_file}")
            
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GROBID full-text extraction on a directory of PDFs."
    )
    parser.add_argument("--input-dir", default="data/input",
                        help="Directory of PDFs (default: data/input)")
    parser.add_argument("--output-dir", default="data/grobid",
                        help="Output directory (default: data/grobid)")
    args = parser.parse_args()

    process_pdfs_in_directory(args.input_dir, args.output_dir)
