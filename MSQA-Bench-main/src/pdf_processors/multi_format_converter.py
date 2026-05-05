import fitz  # PyMuPDF
import os
import markdownify

def extract_text_from_pdf(pdf_path):
    """Extract plain text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    
    # Iterate through each page and extract text
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract as plain text

    return text

def extract_html_from_pdf(pdf_path):
    """Extract HTML format from a PDF file."""
    doc = fitz.open(pdf_path)
    html_content = ""
    
    # Iterate through each page and extract HTML
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        html_content += page.get_text("html")  # Extract as HTML

    return html_content

def convert_to_markdown(text):
    """Convert extracted text to Markdown."""
    markdown_text = markdownify.markdownify(text)
    return markdown_text

def convert_pdfs_in_directory(directory_path):
    """Convert all PDFs in a directory to .txt, .html, and .md."""
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            
            print(f"Processing {filename}...")
            
            # Extract text (plain text)
            text = extract_text_from_pdf(pdf_path)
            text_filename = os.path.join(directory_path, f"{filename}.txt")
            with open(text_filename, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            print(f"  Saved as {text_filename}")
            
            # Extract HTML content
            html_content = extract_html_from_pdf(pdf_path)
            html_filename = os.path.join(directory_path, f"{filename}.html")
            with open(html_filename, "w", encoding="utf-8") as html_file:
                html_file.write(html_content)
            print(f"  Saved as {html_filename}")
            
            # Convert and save as Markdown
            markdown_content = convert_to_markdown(html_content)  # We can use HTML to Markdown
            markdown_filename = os.path.join(directory_path, f"{filename}.md")
            with open(markdown_filename, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
            print(f"  Saved as {markdown_filename}")
    
    print("All PDFs processed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PDFs in a directory to .txt, .html, and .md."
    )
    parser.add_argument("directory", nargs="?", default="data/input",
                        help="Directory of PDFs (default: data/input)")
    args = parser.parse_args()

    convert_pdfs_in_directory(args.directory)
