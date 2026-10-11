import numpy as np, os
ARTEFACT_DIR = r'C:/MLProject/zero_day_project/artefacts_v2'
# List files to find features file
for f in os.listdir(ARTEFACT_DIR):
    if 'feature' in f.lower() or 'select' in f.lower():
        print(f)
