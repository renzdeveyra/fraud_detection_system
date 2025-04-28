# test_imports.py at project root
try:
    from infrastructure.config import load_paths
    print("✅ Imports work!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
