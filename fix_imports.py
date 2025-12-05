"""
Quick fix: Create __init__.py files and verify structure
"""
from pathlib import Path

# Create __init__.py files
folders = ['src', 'config', 'ui', 'tests']

for folder in folders:
    init_file = Path(folder) / "__init__.py"
    init_file.parent.mkdir(exist_ok=True)
    init_file.touch()
    print(f"✅ Created: {init_file}")

print("\n🎉 All __init__.py files created!")
print("\n📝 Next step: Run the app")
print("   uv run main.py")