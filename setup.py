"""
Setup script for RAG Question Classifier
Initializes database and verifies dependencies
"""
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'chromadb': 'chromadb',
        'sqlalchemy': 'sqlalchemy',
        'numpy': 'numpy',
        'pydantic': 'pydantic'
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    return len(missing) == 0, missing


def init_directories():
    """Initialize required directories"""
    print("\nInitializing directories...")
    
    dirs = [
        "data",
        "data/chroma_db",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")


def init_database():
    """Initialize Qdrant database"""
    print("\nInitializing Qdrant database...")
    
    sys.path.insert(0, 'src')
    try:
        from src.database import init_db
        init_db()
        print("  ✓ Qdrant database initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error initializing database: {e}")
        return False


def test_imports():
    """Test if all modules can be imported"""
    print("\nTesting module imports...")
    
    sys.path.insert(0, 'src')
    
    modules_to_test = [
        ('config', 'Configuration'),
        ('classifier', 'Query Classifier'),
        ('vector_db', 'Vector Database'),
        ('database', 'Database Models'),
        ('rag', 'RAG Pipeline'),
        ('utils', 'Utilities')
    ]
    
    all_ok = True
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except Exception as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run setup"""
    print("=" * 60)
    print("RAG Question Classifier - Setup Script")
    print("=" * 60)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    # Initialize directories
    init_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️  Some modules failed to import")
        return False
    
    # Initialize database
    db_ok = init_database()
    
    if not db_ok:
        return False
    
    # Print success message
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read QUICKSTART.md for usage examples")
    print("  2. Run examples/example_1_classification.py")
    print("  3. Run examples/example_2_vector_db.py")
    print("  4. Run examples/example_3_rag_pipeline.py")
    print("\nFor detailed documentation, see README.md")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
