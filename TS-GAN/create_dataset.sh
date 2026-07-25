# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/transition1x_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/transition1x_subset_50/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/transition1x_subset_50/ \
#                     --output_path ./data/transition1x_subset_50_numpy/train/train.py

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/isomer_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/isomer_subset_50/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/isomer_subset_50/ \
#                     --output_path ./data/isomer_subset_50_numpy/train/train.py

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgd1_subset_50_splits/train_split.pkl \
# #                     --output_dir ./data/rgd1_subset_50/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgd1_subset_50/ \
#                     --output_path ./data/rgd1_subset_50_numpy/train/train.py

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/transition1x_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/transition1x_subset_10/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/transition1x_subset_10/ \
#                     --output_path ./data/transition1x_subset_10_numpy/train/train.npy

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/isomer_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/isomer_subset_10/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/isomer_subset_10/ \
#                     --output_path ./data/isomer_subset_10_numpy/train/train.npy

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgd1_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgd1_subset_10/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgd1_subset_10/ \
#                     --output_path ./data/rgd1_subset_10_numpy/train/train.npy

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgdx_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgdx_subset_10/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgdx_subset_10/ \
#                     --output_path ./data/rgdx_subset_10_numpy/train/train.npy


# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgdy_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgdy_subset_10/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgdy_subset_10/ \
#                     --output_path ./data/rgdy_subset_10_numpy/train/train.npy

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgdx_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/rgdx_subset_50/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgdx_subset_50/ \
#                     --output_path ./data/rgdx_subset_50_numpy/train/train.npy

# sbatch script_a40.slurm oa2 convert_pkl_to_xyx.py --input_pkl ../ts_project/OAReactDiff/oa_reactdiff/data/rgdy_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/rgdy_subset_50/
# sbatch script_a40.slurm oa2 gen_data.py --xyz_dir ./data/rgdy_subset_50/ \
#                     --output_path ./data/rgdy_subset_50_numpy/train/train.npy

# sbatch script_l40.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl /home/rbs-ml/205030033/ankit/permuted_data/transition1x_splits/test_split.pkl \
#                     --output_dir /home/rbs-ml/205030033/ankit/permuted_data/transition1x_splits/xyz_data/

# sbatch script_l40.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl /home/rbs-ml/205030033/ankit/permuted_data/isomer_splits/test_split.pkl \
#                     --output_dir /home/rbs-ml/205030033/ankit/permuted_data/isomer_splits/xyz_data/

# sbatch script_l40.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl /home/rbs-ml/205030033/ankit/permuted_data/rgd1_splits/test_split.pkl \
#                     --output_dir /home/rbs-ml/205030033/ankit/permuted_data/rgd1_splits/xyz_data/

sbatch script_l40.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl /home/rbs-ml/205030033/ankit/permuted_data/rgdx_splits/test_split.pkl \
                    --output_dir /home/rbs-ml/205030033/ankit/permuted_data/rgdx_splits/xyz_data/

# sbatch script_dgx.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl /home/rbs-ml/205030033/ankit/permuted_data/rgdy_splits/test_split.pkl \
#                     --output_dir /home/rbs-ml/205030033/ankit/permuted_data/rgdy_splits/xyz_data/