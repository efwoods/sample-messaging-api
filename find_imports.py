# filename: find_imports.py
import os
import sys
import pkgutil
import stdlib_list

# Get standard library for your Python version
stdlib = set(stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))

imports = set()

# Walk through the project and collect imports
for root, _, files in os.walk("app"):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("import "):
                        parts = line.split()
                        if len(parts) > 1:
                            imports.add(parts[1].split(".")[0])
                    elif line.startswith("from "):
                        parts = line.split()
                        if len(parts) > 1:
                            imports.add(parts[1].split(".")[0])

# Filter out standard library
third_party = sorted(i for i in imports if i not in stdlib)

print("=== Third-party imports (candidates for requirements.txt) ===")
for lib in third_party:
    print(lib)
