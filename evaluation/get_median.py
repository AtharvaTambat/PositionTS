import numpy as np
import sys
import re
import ast

logfile = sys.argv[1]

def extract_all(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    data = {}

    # Extract lists
    patterns = {
        "delta_Es": r"delta Es:\s*(\[[^\]]*\])",
        "rmsds": r"rmsds:\s*(\[[^\]]*\])",
        "dmaes": r"dmaes:\s*(\[[^\]]*\])",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        data[key] = ast.literal_eval(match.group(1)) if match else None

    path_match = re.search(r"(/home/[^\n]+)", text)
    data["path"] = path_match.group(1) if path_match else None

    return data

data = extract_all(logfile)

delta_Es = np.array(data["delta_Es"])
rmsds = np.array(data["rmsds"])
dmaes = np.array(data["dmaes"])

median_rmsd = np.median(rmsds)
median_dmae = np.median(dmaes)
median_delE = np.median(delta_Es)

print("\n")
print(data["path"])
print(f"Median RMSD: {np.round(median_rmsd, 3)}")
print(f"Median D-MAE: {np.round(median_dmae, 3)}")
print(f"Median MAE: {np.round(median_delE, 3)}\n")
# print(np.median(delta_Es), np.median(rmsds), np.median(dmaes))
# print(np.mean(delta_Es), np.mean(rmsds), np.mean(dmaes))
# print(delta_Es, rmsds, dmaes)