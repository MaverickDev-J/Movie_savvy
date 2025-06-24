#!/usr/bin/env python3
"""
Script to inspect what's available in the LitGPT installation
"""

import sys
from pathlib import Path

# Add LitGPT to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
LITGPT_DIR = BASE_DIR.parent / "llm-finetune" / "litgpt"
sys.path.insert(0, str(LITGPT_DIR))

print(f"LitGPT directory: {LITGPT_DIR}")
print(f"LitGPT exists: {LITGPT_DIR.exists()}")

try:
    import litgpt
    print(f"LitGPT module location: {litgpt.__file__}")
    print(f"LitGPT version: {getattr(litgpt, '__version__', 'unknown')}")
    
    # List all available attributes
    print("\nAvailable in litgpt:")
    for attr in sorted(dir(litgpt)):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # Try specific imports
    print("\nTrying specific imports:")
    
    try:
        from litgpt.model import GPT
        print("✓ GPT class available")
    except ImportError as e:
        print(f"✗ GPT import failed: {e}")
    
    try:
        from litgpt.config import Config
        print("✓ Config class available")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
    
    try:
        from litgpt.tokenizer import Tokenizer
        print("✓ Tokenizer class available")
    except ImportError as e:
        print(f"✗ Tokenizer import failed: {e}")
    
    try:
        from litgpt.generate.base import generate
        print("✓ generate function available")
    except ImportError as e:
        try:
            from litgpt.generate import generate
            print("✓ generate function available (alternative path)")
        except ImportError as e2:
            print(f"✗ generate import failed: {e2}")
    
    try:
        from litgpt.utils import load_checkpoint
        print("✓ load_checkpoint available")
    except ImportError as e:
        print(f"✗ load_checkpoint import failed: {e}")
    
    # Check for CLI commands
    print("\nChecking for CLI commands:")
    try:
        from litgpt.chat.base import main as chat_main
        print("✓ Chat CLI available")
    except ImportError:
        try:
            from litgpt.chat import main as chat_main
            print("✓ Chat CLI available (alternative)")
        except ImportError as e:
            print(f"✗ Chat CLI not found: {e}")

except ImportError as e:
    print(f"Failed to import litgpt: {e}")

# Check model directory structure
MODEL_DIR = BASE_DIR / "models" / "mistral-7b-finetuned"
print(f"\nModel directory: {MODEL_DIR}")
print(f"Model directory exists: {MODEL_DIR.exists()}")

if MODEL_DIR.exists():
    print("Files in model directory:")
    for file in MODEL_DIR.iterdir():
        print(f"  - {file.name}")