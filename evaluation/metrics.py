import numpy as np
import sys
import re
import ast

logfile1 = sys.argv[1]
logfile2 = sys.argv[2]

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

data1 = extract_all(logfile1)
data2 = extract_all(logfile2)

print("\n")
print(data1["path"])
delta_Es_r = np.array(data1["delta_Es"])
rmsds_r = np.array(data1["rmsds"])
dmaes_r = np.array(data1["dmaes"])

delta_Es_p = np.array(data2["delta_Es"])
rmsds_p = np.array(data2["rmsds"])
dmaes_p = np.array(data2["dmaes"])

data_E = np.concatenate([delta_Es_r[:, np.newaxis], delta_Es_p[:, np.newaxis]], axis=1)
min_data_E = np.min(data_E, axis=1)

data_rmsd = np.concatenate([rmsds_r[:, np.newaxis], rmsds_p[:, np.newaxis]], axis=1)
min_data_rmsd = np.min(data_rmsd, axis=1)

data_dmae = np.concatenate([dmaes_r[:, np.newaxis], dmaes_p[:, np.newaxis]], axis=1)
min_data_dmae = np.min(data_dmae, axis=1)

##### Reactant side #####
mean_rmsd_r = np.mean(rmsds_r)
std_rmsd_r = np.std(rmsds_r)
median_rmsd_r = np.median(rmsds_r)

mean_dmae_r = np.mean(dmaes_r)
std_dmae_r = np.std(dmaes_r)
median_dmae_r = np.median(dmaes_r)

rmse_r = np.sqrt(np.mean(delta_Es_r**2))
median_delE_r = np.median(delta_Es_r)

print("Reactant Side: \n")
print(f"RMSD: {np.round(mean_rmsd_r,3)}$\pm${np.round(std_rmsd_r,3)}")
print(f"DMAE: {np.round(mean_dmae_r,3)}$\pm${np.round(std_dmae_r,3)}")
print(f"RMSE: {np.round(rmse_r,3)}")
print(f"Median RMSD: {np.round(median_rmsd_r, 3)}")
print(f"Median D-MAE: {np.round(median_dmae_r, 3)}")
print(f"Median MAE: {np.round(median_delE_r, 3)}\n")

##### Product side #####
mean_rmsd_p = np.mean(rmsds_p)
std_rmsd_p = np.std(rmsds_p)
median_rmsd_p = np.median(rmsds_p)

mean_dmae_p = np.mean(dmaes_p)
std_dmae_p = np.std(dmaes_p)
median_dmae_p = np.median(dmaes_p)

rmse_p = np.sqrt(np.mean(delta_Es_p**2))
median_delE_p = np.median(delta_Es_p)

print("Product Side: \n")
print(f"RMSD: {np.round(mean_rmsd_p,3)}$\pm${np.round(std_rmsd_p,3)}")
print(f"DMAE: {np.round(mean_dmae_p,3)}$\pm${np.round(std_dmae_p,3)}")
print(f"RMSE: {np.round(rmse_p,3)}")
print(f"Median RMSD: {np.round(median_rmsd_p, 3)}")
print(f"Median D-MAE: {np.round(median_dmae_p, 3)}")
print(f"Median MAE: {np.round(median_delE_p, 3)}\n")


##### Minimum side #####
mean_rmsd_min = np.mean(min_data_rmsd)
std_rmsd_min = np.std(min_data_rmsd)
median_rmsd_min = np.median(min_data_rmsd)

mean_dmae_min = np.mean(min_data_dmae)
std_dmae_min = np.std(min_data_dmae)
median_dmae_min = np.median(min_data_dmae)

rmse_min = np.sqrt(np.mean(min_data_E**2))
median_delE_min = np.median(min_data_E)

print("Minimum Side: \n")
print(f"RMSD: {np.round(mean_rmsd_min,3)}$\pm${np.round(std_rmsd_min,3)}")
print(f"DMAE: {np.round(mean_dmae_min,3)}$\pm${np.round(std_dmae_min,3)}")
print(f"RMSE: {np.round(rmse_min,3)}")
print(f"Median RMSD: {np.round(median_rmsd_min, 3)}")
print(f"Median D-MAE: {np.round(median_dmae_min, 3)}")
print(f"Median MAE: {np.round(median_delE_min, 3)}\n")

# median_rmsd = np.median(rmsds)
# median_dmae = np.median(dmaes)
# median_delE = np.median(delta_Es)

# print("\n")
# print(data["path"])
# print(f"Median RMSD: {np.round(median_rmsd, 3)}")
# print(f"Median D-MAE: {np.round(median_dmae, 3)}")
# print(f"Median MAE: {np.round(median_delE, 3)}\n")
# # print(np.median(delta_Es), np.median(rmsds), np.median(dmaes))
# # print(np.mean(delta_Es), np.mean(rmsds), np.mean(dmaes))
# # print(delta_Es, rmsds, dmaes)