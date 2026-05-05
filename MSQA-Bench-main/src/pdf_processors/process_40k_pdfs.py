#!/usr/bin/env python3
"""
Bulk PDF Processing Script for 40,000 Files

This script is optimized for processing very large batches of PDFs with:
- Automatic resume on failure
- OCR fallback for scanned documents
- Progress tracking and reporting
- Resource management
"""

import sys
import json
import logging
import signal
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import psutil
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vision_extractors.fast_pdf_extractor import FastPDFExtractor


class BulkPDFProcessor:
    """Optimized processor for very large PDF batches."""
    
    def __init__(self, config_path: str = "config/fast_extractor_config.json"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging
        self._setup_logging()
        
        # Create extractor
        self.extractor = self._create_extractor()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_config = self.config['logging']
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup rotating file handler
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(
            log_config['log_file'],
            maxBytes=log_config.get('rotate_size_mb', 100) * 1024 * 1024,
            backupCount=log_config.get('backup_count', 5)
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        logger.addHandler(handler)
        
        # Also log to console
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        self.logger = logger
    
    def _create_extractor(self) -> FastPDFExtractor:
        """Create the PDF extractor with config settings."""
        ext_config = self.config['extractor']
        perf_config = self.config['performance']
        
        return FastPDFExtractor(
            output_dir=ext_config['output_dir'],
            log_file=ext_config['log_file'],
            progress_file=ext_config['progress_file'],
            min_text_length=ext_config['min_text_length'],
            ocr_threshold=ext_config['ocr_threshold'],
            max_workers=perf_config['max_workers'],
            use_ocr_fallback=ext_config['use_ocr_fallback'],
            clean_text=ext_config['clean_text'],
            save_as_markdown=ext_config['save_as_markdown']
        )
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.warning(f"Received signal {signum}. Shutting down gracefully...")
        self.interrupted = True
        
        # Save current progress
        if hasattr(self, 'extractor'):
            self.extractor._save_progress()
            self.logger.info("Progress saved. You can resume by running the script again.")
    
    def _check_resources(self) -> bool:
        """Check if system has enough resources."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self.logger.info(f"System resources:")
        self.logger.info(f"  - CPU usage: {cpu_percent}%")
        self.logger.info(f"  - Memory available: {memory.available / (1024**3):.1f} GB")
        self.logger.info(f"  - Memory usage: {memory.percent}%")
        
        if memory.percent > 90:
            self.logger.warning("Memory usage is very high. Consider reducing max_workers.")
            return False
        
        return True
    
    def _set_process_priority(self):
        """Set process priority for background operation."""
        try:
            resource_config = self.config.get('resource_management', {})
            
            # Set nice level (lower priority)
            nice_level = resource_config.get('nice_level', 10)
            os.nice(nice_level)
            self.logger.info(f"Set process nice level to {nice_level}")
            
            # Set IO nice if available (Linux only)
            if sys.platform == 'linux':
                try:
                    import subprocess
                    io_class = resource_config.get('io_nice_class', 2)
                    io_level = resource_config.get('io_nice_level', 4)
                    subprocess.run(
                        ['ionice', '-c', str(io_class), '-n', str(io_level), 
                         '-p', str(os.getpid())],
                        check=True
                    )
                    self.logger.info(f"Set IO nice to class {io_class}, level {io_level}")
                except Exception as e:
                    self.logger.debug(f"Could not set IO nice: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Could not set process priority: {e}")
    
    def process_all(self) -> dict:
        """Process all PDFs with the configured settings."""
        self.start_time = datetime.now()
        
        # Check resources
        if not self._check_resources():
            self.logger.error("Insufficient resources. Exiting.")
            return {"error": "Insufficient resources"}
        
        # Set process priority for background operation
        self._set_process_priority()
        
        batch_config = self.config['batch_processing']
        error_config = self.config['error_handling']
        
        # Count total PDFs
        input_dir = Path(batch_config['input_dir'])
        pattern = batch_config['pattern']
        
        if not input_dir.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return {"error": "Input directory not found"}
        
        pdf_files = list(input_dir.glob(pattern))
        total_files = len(pdf_files)
        
        self.logger.info("=" * 70)
        self.logger.info(f"BULK PDF PROCESSING - {total_files:,} FILES")
        self.logger.info("=" * 70)
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.config['extractor']['output_dir']}")
        self.logger.info(f"Workers: {self.config['performance']['max_workers']}")
        self.logger.info(f"OCR fallback: {self.config['extractor']['use_ocr_fallback']}")
        self.logger.info("=" * 70)
        
        # Check existing progress
        if self.extractor.progress.get('completed'):
            completed_count = len(self.extractor.progress['completed'])
            self.logger.info(f"Found {completed_count:,} already completed files")
            self.logger.info(f"Remaining: {total_files - completed_count:,} files")
        
        # Start processing
        self.logger.info("Starting batch extraction...")
        self.logger.info("Press Ctrl+C to interrupt (progress will be saved)")
        
        try:
            results = self.extractor.batch_extract(
                input_dir=str(input_dir),
                pattern=pattern,
                force=batch_config['force_reprocess'],
                parallel=batch_config['parallel']
            )
            
            # Process results
            stats = results.get('stats', {})
            
            # Save failed PDFs list if any
            if stats.get('failed', 0) > 0 and error_config.get('save_failed_list'):
                failed_list = [
                    r['pdf_path'] for r in results.get('results', [])
                    if not r.get('success')
                ]
                
                failed_file = Path(error_config['save_failed_list'])
                failed_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(failed_file, 'w') as f:
                    json.dump(failed_list, f, indent=2)
                
                self.logger.info(f"Failed PDFs list saved to: {failed_file}")
            
            # Calculate and log final statistics
            self._log_final_stats(stats, total_files)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            # Always save progress
            self.extractor._save_progress()
    
    def _log_final_stats(self, stats: dict, total_files: int):
        """Log comprehensive final statistics."""
        elapsed = datetime.now() - self.start_time
        
        self.logger.info("=" * 70)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total files: {total_files:,}")
        self.logger.info(f"Successful: {stats.get('successful', 0):,}")
        self.logger.info(f"Failed: {stats.get('failed', 0):,}")
        self.logger.info(f"Skipped: {stats.get('skipped', 0):,}")
        
        if stats.get('successful', 0) > 0:
            self.logger.info(f"\nPerformance:")
            self.logger.info(f"  - Total time: {elapsed}")
            self.logger.info(f"  - Pages extracted: {stats.get('total_pages', 0):,}")
            self.logger.info(f"  - Avg pages/file: {stats.get('avg_pages_per_file', 0):.1f}")
            self.logger.info(f"  - Avg time/file: {stats.get('avg_time_per_file', 0):.2f}s")
            
            # Calculate overall speed
            files_per_hour = (stats['successful'] / elapsed.total_seconds()) * 3600
            self.logger.info(f"  - Processing speed: {files_per_hour:.0f} files/hour")
            
            # Estimate remaining time if not complete
            remaining = total_files - stats['successful'] - stats.get('failed', 0) - stats.get('skipped', 0)
            if remaining > 0:
                est_hours = remaining / files_per_hour
                self.logger.info(f"\nEstimated time for remaining {remaining:,} files: {est_hours:.1f} hours")
        
        self.logger.info("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process 40,000 PDFs with OCR fallback"
    )
    parser.add_argument(
        "--config",
        default="config/fast_extractor_config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check configuration and resources"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = BulkPDFProcessor(args.config)
    
    if args.check_only:
        processor._check_resources()
        print("\nConfiguration loaded successfully.")
        print(f"Ready to process PDFs from: {processor.config['batch_processing']['input_dir']}")
        return
    
    # Process all PDFs
    results = processor.process_all()
    
    # Exit with appropriate code
    if results.get('error'):
        sys.exit(1)
    elif results.get('stats', {}).get('failed', 0) > 0:
        sys.exit(2)  # Some failures
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()

