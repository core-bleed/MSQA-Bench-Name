"""Two-column / one-column PDF text extractor.

Splits each page into a left and right half, extracts text from each side,
then trims content to the span between the first abstract/introduction
heading and the references heading.
"""

import argparse
from pathlib import Path

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            bbox = page.bbox
            x_start, y_start = bbox[0], bbox[1]

            left_page = page.crop((x_start, y_start, page.width / 2, page.height))
            right_page = page.crop((page.width / 2, y_start, page.width, page.height))

            left_text = left_page.extract_text()
            right_text = right_page.extract_text()

            if left_text:
                left_text = " ".join(left_text.split())
            if right_text:
                right_text = " ".join(right_text.split())

            text += f"{left_text} {right_text}"

        text = text.replace("\n", " ")

        start_candidates = [i for i in [text.lower().find(x) for x in ("abstract", "introduction")] if i != -1]
        end_candidates = [i for i in [text.lower().find("references")] if i != -1]
        if start_candidates and end_candidates:
            start_index = min(start_candidates)
            end_index = min(end_candidates)
            if start_index < end_index:
                text = text[start_index:end_index]

        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output text file (default: stdout)")
    args = parser.parse_args()

    text = extract_text_from_pdf(str(args.pdf))
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
