import os
import glob

root_dir = r"D:\Vyapti"

# Files to process
extensions = ['*.py', '*.md', '*.toml', '*.txt', '*.json']

# Replacements (Order matters! More specific first)
replacements = [
    ("VyaptiEnv", "VyaptiEnv"),
    ("Vyapti", "Vyapti"),
    ("vyapti", "vyapti")
]

modified_count = 0

for ext in extensions:
    for filepath in glob.glob(os.path.join(root_dir, '**', ext), recursive=True):
        # Skip git and cache directories
        if ".git" in filepath or "__pycache__" in filepath:
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue
        
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
            
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            modified_count += 1

# Rename files if any
rename_count = 0
for root, dirs, files in os.walk(root_dir):
    for name in files:
        if "vyapti" in name.lower():
            old_path = os.path.join(root, name)
            new_name = name.replace("Vyapti", "Vyapti").replace("vyapti", "vyapti")
            new_path = os.path.join(root, new_name)
            os.rename(old_path, new_path)
            rename_count += 1

print(f"Refactor complete. Modified {modified_count} files, renamed {rename_count} files.")
