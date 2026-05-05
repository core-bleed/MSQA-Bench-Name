#!/usr/bin/env python3
"""
Example: Using the Agentic PDF Vision Extractor

This script demonstrates how to use the agentic extractor programmatically
with custom configurations and quality monitoring.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision_extractors.agentic_vision_extractor import (
    AgenticPDFVisionExtractor
)


def example_single_file():
    """Extract text from a single PDF with quality monitoring."""
    print("="*60)
    print("Example 1: Single File Extraction")
    print("="*60)
    
    # Initialize extractor
    extractor = AgenticPDFVisionExtractor(
        config_path="config/config.json",
        model="llama3.2-vision:latest",
        output_dir="extracted_text/examples",
        log_level="INFO"
    )
    
    # Extract from single file
    pdf_path = "data/input/0033d41e726ea6613508477b926cbcd76d2c499e.pdf"
    
    if Path(pdf_path).exists():
        success = extractor.extract_from_pdf(
            pdf_path,
            output_filename="example_output.txt",
            start_page=1,
            end_page=5  # Process first 5 pages only
        )
        
        if success:
            print("\n✓ Extraction completed successfully!")
            
            # Check quality scores
            progress_key = extractor._get_file_hash(Path(pdf_path))
            if progress_key in extractor.progress:
                quality_scores = extractor.progress[progress_key].get(
                    'quality_scores', {}
                )
                print("\nQuality Scores by Page:")
                for page, score in sorted(quality_scores.items()):
                    status = "✓" if float(score) >= 0.6 else "⚠"
                    print(f"  Page {page}: {status} {score:.2f}")
        else:
            print("\n✗ Extraction failed!")
    else:
        print(f"PDF not found: {pdf_path}")


def example_batch_with_monitoring():
    """Batch extract with quality monitoring and reporting."""
    print("\n" + "="*60)
    print("Example 2: Batch Extraction with Quality Monitoring")
    print("="*60)
    
    extractor = AgenticPDFVisionExtractor(
        config_path="config/config.json",
        output_dir="extracted_text/examples/batch"
    )
    
    input_dir = "data/input"
    
    if Path(input_dir).exists():
        # Run batch extraction
        results = extractor.batch_extract(input_dir, resume=True)
        
        # Analyze results
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        
        successful = sum(results.values())
        total = len(results)
        
        print(f"Total files: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        # Quality analysis
        print("\n" + "="*60)
        print("QUALITY ANALYSIS")
        print("="*60)
        
        all_qualities = []
        for progress_key, file_data in extractor.progress.items():
            if file_data.get('completed'):
                avg_quality = file_data.get('average_quality', 0.0)
                filename = Path(file_data['file_path']).name
                all_qualities.append((filename, avg_quality))
        
        # Sort by quality
        all_qualities.sort(key=lambda x: x[1])
        
        print("\nLowest Quality Files (need review):")
        for filename, quality in all_qualities[:5]:
            print(f"  {quality:.2f} - {filename}")
        
        print("\nHighest Quality Files:")
        for filename, quality in all_qualities[-5:]:
            print(f"  {quality:.2f} - {filename}")
    else:
        print(f"Input directory not found: {input_dir}")


def example_custom_config():
    """Use custom agentic configuration for difficult PDFs."""
    print("\n" + "="*60)
    print("Example 3: Custom Config for Difficult PDFs")
    print("="*60)
    
    # Create custom config (more aggressive retry)
    custom_config = {
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama3.2-vision:latest",
            "timeout": 600,  # Longer timeout
            "temperature": 0.0,  # Start deterministic
            "top_p": 0.9
        },
        "processing": {
            "image_dpi": 300,  # Higher DPI for better quality
            "output_dir": "extracted_text/examples/difficult",
            "input_dir": "data/input",
            "log_level": "DEBUG"
        },
        "logging": {
            "log_file": "logs/difficult_extraction.log"
        },
        "agentic": {
            "max_retries": 5,  # More retries
            "quality_threshold": 0.5,  # Lower threshold
            "min_text_length": 30,  # Accept shorter text
            "max_repetition_ratio": 0.25,  # Stricter repetition check
            "enable_reflection": True
        },
        "text_cleaning": {
            "apply_unidecode": True,
            "remove_extra_whitespace": True,
            "min_text_length": 10
        }
    }
    
    # Save to temporary config file
    import json
    temp_config = Path("temp_config.json")
    with open(temp_config, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    try:
        extractor = AgenticPDFVisionExtractor(
            config_path=str(temp_config)
        )
        
        print("\nCustom configuration loaded:")
        print(f"  Max retries: {extractor.max_retries}")
        print(f"  Quality threshold: {extractor.quality_threshold}")
        print(f"  Image DPI: {extractor.image_dpi}")
        print(f"  Temperature: {extractor.temperature}")
        
        # Example: extract from a difficult PDF
        # (Replace with actual difficult PDF path)
        print("\nReady to process difficult PDFs with aggressive retry...")
        
    finally:
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()


def example_quality_filtering():
    """Filter and re-extract low-quality pages."""
    print("\n" + "="*60)
    print("Example 4: Quality-Based Re-extraction")
    print("="*60)
    
    extractor = AgenticPDFVisionExtractor(
        config_path="config/config.json"
    )
    
    # Find low-quality pages from progress
    low_quality_pages = []
    
    for progress_key, file_data in extractor.progress.items():
        if not file_data.get('completed'):
            continue
        
        quality_scores = file_data.get('quality_scores', {})
        file_path = file_data.get('file_path')
        
        for page, score in quality_scores.items():
            if float(score) < 0.6:  # Below threshold
                low_quality_pages.append({
                    'file': Path(file_path).name,
                    'page': int(page),
                    'score': float(score),
                    'path': file_path
                })
    
    if low_quality_pages:
        print(f"\nFound {len(low_quality_pages)} low-quality pages")
        print("\nLowest quality pages:")
        
        # Sort by score
        low_quality_pages.sort(key=lambda x: x['score'])
        
        for item in low_quality_pages[:10]:
            print(
                f"  {item['score']:.2f} - {item['file']} "
                f"(page {item['page']})"
            )
        
        # Option to re-extract
        print("\nTo re-extract these pages:")
        print("1. Delete their entries from progress file")
        print("2. Run extraction again with --no-resume")
        print("3. Or extract specific pages with --start-page/--end-page")
    else:
        print("\n✓ All extracted pages meet quality threshold!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AGENTIC PDF VISION EXTRACTOR - EXAMPLES")
    print("="*60)
    
    # Check if Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
        else:
            print("⚠ Ollama may not be running properly")
    except Exception:
        print("✗ Ollama is not running. Start it with: ollama serve")
        return
    
    # Run examples
    try:
        example_single_file()
        # example_batch_with_monitoring()
        # example_custom_config()
        # example_quality_filtering()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

