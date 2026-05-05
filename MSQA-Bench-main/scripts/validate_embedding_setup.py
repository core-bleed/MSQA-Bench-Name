#!/usr/bin/env python3
"""
Validation script to check if the environment is properly set up
for embedding model fine-tuning.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple


class SetupValidator:
    """Validate the setup for embedding model training."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
    
    def print_header(self, text: str) -> None:
        """Print a section header."""
        print("\n" + "=" * 70)
        print(text)
        print("=" * 70)
    
    def print_check(self, name: str, passed: bool, message: str = "") -> None:
        """Print check result."""
        status = "✓" if passed else "✗"
        status_text = "PASS" if passed else "FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        
        print(f"{color}{status} {name}: {status_text}{reset}")
        if message:
            print(f"  {message}")
        
        if passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
    
    def print_warning(self, name: str, message: str) -> None:
        """Print warning."""
        print(f"\033[93m⚠ {name}: WARNING\033[0m")
        print(f"  {message}")
        self.warnings += 1
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 8
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if is_valid:
            self.print_check(
                "Python Version",
                True,
                f"Python {version_str} (>= 3.8 required)"
            )
        else:
            self.print_check(
                "Python Version",
                False,
                f"Python {version_str} found, but 3.8+ required"
            )
        
        return is_valid
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if required packages are installed."""
        required_packages = {
            "torch": "PyTorch",
            "sentence_transformers": "Sentence Transformers",
            "transformers": "HuggingFace Transformers",
            "numpy": "NumPy",
        }
        
        missing = []
        installed = []
        
        for package, name in required_packages.items():
            try:
                __import__(package)
                installed.append(name)
            except ImportError:
                missing.append(name)
        
        all_installed = len(missing) == 0
        
        if all_installed:
            self.print_check(
                "Dependencies",
                True,
                f"All required packages installed: {', '.join(installed)}"
            )
        else:
            self.print_check(
                "Dependencies",
                False,
                f"Missing packages: {', '.join(missing)}"
            )
            print("  Install with: pip install -r requirements.txt")
        
        return all_installed, missing
    
    def check_gpu_availability(self) -> None:
        """Check if GPU is available."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.print_check(
                    "GPU Availability",
                    True,
                    f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)"
                )
            else:
                self.print_warning(
                    "GPU Availability",
                    "No GPU detected. Training will use CPU (much slower)."
                )
        except Exception as e:
            self.print_warning(
                "GPU Check",
                f"Could not check GPU: {e}"
            )
    
    def check_file_exists(self, path: Path, description: str) -> bool:
        """Check if a file exists."""
        exists = path.exists()
        
        if exists:
            if path.is_file():
                size = path.stat().st_size
                size_str = self._format_size(size)
                self.print_check(
                    description,
                    True,
                    f"Found: {path} ({size_str})"
                )
            else:
                self.print_check(
                    description,
                    True,
                    f"Found: {path}"
                )
        else:
            self.print_check(
                description,
                False,
                f"Not found: {path}"
            )
        
        return exists
    
    def check_jsonl_format(self, path: Path) -> bool:
        """Check if JSONL file is properly formatted."""
        if not path.exists():
            return False
        
        try:
            valid_count = 0
            total_lines = 0
            
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_lines += 1
                    try:
                        record = json.loads(line)
                        if "question" in record and "answer" in record:
                            valid_count += 1
                    except json.JSONDecodeError:
                        pass
            
            if valid_count > 0:
                self.print_check(
                    "JSONL Format",
                    True,
                    f"{valid_count}/{total_lines} valid Q&A pairs found"
                )
                return True
            else:
                self.print_check(
                    "JSONL Format",
                    False,
                    f"No valid Q&A pairs found in {total_lines} lines"
                )
                return False
                
        except Exception as e:
            self.print_check(
                "JSONL Format",
                False,
                f"Error reading file: {e}"
            )
            return False
    
    def check_config_file(self, path: Path) -> bool:
        """Check if config file is valid."""
        if not path.exists():
            return False
        
        try:
            with path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            
            required_keys = [
                "input_jsonl",
                "output_dir",
                "base_model",
                "num_epochs",
                "train_batch_size"
            ]
            
            missing_keys = [key for key in required_keys if key not in config]
            
            if not missing_keys:
                self.print_check(
                    "Config File",
                    True,
                    f"Valid configuration with all required keys"
                )
                return True
            else:
                self.print_check(
                    "Config File",
                    False,
                    f"Missing keys: {', '.join(missing_keys)}"
                )
                return False
                
        except json.JSONDecodeError as e:
            self.print_check(
                "Config File",
                False,
                f"Invalid JSON: {e}"
            )
            return False
        except Exception as e:
            self.print_check(
                "Config File",
                False,
                f"Error reading file: {e}"
            )
            return False
    
    def check_disk_space(self, required_gb: float = 5.0) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil
            
            stats = shutil.disk_usage(".")
            free_gb = stats.free / (1024 ** 3)
            
            if free_gb >= required_gb:
                self.print_check(
                    "Disk Space",
                    True,
                    f"{free_gb:.1f} GB available (>= {required_gb} GB required)"
                )
                return True
            else:
                self.print_check(
                    "Disk Space",
                    False,
                    f"Only {free_gb:.1f} GB available (>= {required_gb} GB required)"
                )
                return False
        except Exception as e:
            self.print_warning(
                "Disk Space",
                f"Could not check disk space: {e}"
            )
            return True
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        self.print_header("Environment Setup Validation")
        
        # System checks
        self.check_python_version()
        deps_ok, missing = self.check_dependencies()
        self.check_gpu_availability()
        self.check_disk_space()
        
        # File structure checks
        self.print_header("File Structure")
        
        config_file = Path("config/embedding_finetuner.json")
        self.check_file_exists(config_file, "Config File")
        if config_file.exists():
            self.check_config_file(config_file)
        
        input_jsonl = Path("data/consolidated_qa.jsonl")
        if config_file.exists():
            try:
                with config_file.open("r", encoding="utf-8") as f:
                    input_jsonl = Path(json.load(f).get("input_jsonl", str(input_jsonl)))
            except Exception:
                pass
        if self.check_file_exists(input_jsonl, "Input JSONL"):
            self.check_jsonl_format(input_jsonl)
        
        self.check_file_exists(
            Path("src/embedding_trainers/embedding_finetuner.py"),
            "Fine-tuner Script"
        )
        
        self.check_file_exists(
            Path("examples/use_finetuned_embeddings.py"),
            "Example Script"
        )
        
        self.check_file_exists(
            Path("scripts/train_embeddings.sh"),
            "Training Script"
        )
        
        # Summary
        self.print_header("Validation Summary")
        
        print(f"\nChecks passed:  {self.checks_passed}")
        print(f"Checks failed:  {self.checks_failed}")
        print(f"Warnings:       {self.warnings}")
        
        if self.checks_failed == 0:
            print("\n\033[92m✓ Environment is ready for training!\033[0m")
            print("\nNext steps:")
            print("  1. Review config: config/embedding_finetuner.json")
            print("  2. Start training: ./scripts/train_embeddings.sh")
            print("  3. Or use Python: python src/embedding_trainers/embedding_finetuner.py")
            return True
        else:
            print("\n\033[91m✗ Please fix the issues above before training.\033[0m")
            
            if not deps_ok:
                print("\nTo install dependencies:")
                print("  pip install -r requirements.txt")
            
            if not input_jsonl.exists():
                print("\nTo prepare Q&A data:")
                print("  python scripts/consolidate_qa.py")
                print("  # or download MSQA-Bench splits from Hugging Face")
            
            return False


def main():
    """Run validation checks."""
    validator = SetupValidator()
    success = validator.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
