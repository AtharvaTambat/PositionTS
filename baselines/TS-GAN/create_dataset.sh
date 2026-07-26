# Set these to your local paths before running.
DATA_ROOT="${DATA_ROOT:-../../data}"                 # dataset splits (shared with the OA-ReactDiff experiments)
PERMUTED_ROOT="${PERMUTED_ROOT:-../../data/permuted}"  # atom-order-permuted copies of the splits
MODEL_ROOT="${MODEL_ROOT:-./trained_models}"      # trained checkpoints
OUTPUT_ROOT="${OUTPUT_ROOT:-./generated}"         # generated transition states

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/transition1x_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/transition1x_subset_50/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/transition1x_subset_50/ \
#                     --output_path ./data/transition1x_subset_50_numpy/train/train.py

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/isomer_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/isomer_subset_50/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/isomer_subset_50/ \
#                     --output_path ./data/isomer_subset_50_numpy/train/train.py

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgd1_subset_50_splits/train_split.pkl \
# #                     --output_dir ./data/rgd1_subset_50/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgd1_subset_50/ \
#                     --output_path ./data/rgd1_subset_50_numpy/train/train.py

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/transition1x_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/transition1x_subset_10/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/transition1x_subset_10/ \
#                     --output_path ./data/transition1x_subset_10_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/isomer_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/isomer_subset_10/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/isomer_subset_10/ \
#                     --output_path ./data/isomer_subset_10_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgd1_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgd1_subset_10/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgd1_subset_10/ \
#                     --output_path ./data/rgd1_subset_10_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgdx_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgdx_subset_10/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgdx_subset_10/ \
#                     --output_path ./data/rgdx_subset_10_numpy/train/train.npy


# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgdy_subset_10_splits/train_split.pkl \
#                     --output_dir ./data/rgdy_subset_10/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgdy_subset_10/ \
#                     --output_path ./data/rgdy_subset_10_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgdx_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/rgdx_subset_50/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgdx_subset_50/ \
#                     --output_path ./data/rgdx_subset_50_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx.py --input_pkl $DATA_ROOT/rgdy_subset_50_splits/train_split.pkl \
#                     --output_dir ./data/rgdy_subset_50/
# sbatch run.slurm oa2 gen_data.py --xyz_dir ./data/rgdy_subset_50/ \
#                     --output_path ./data/rgdy_subset_50_numpy/train/train.npy

# sbatch run.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl $PERMUTED_ROOT/transition1x_splits/test_split.pkl \
#                     --output_dir $PERMUTED_ROOT/transition1x_splits/xyz_data/

# sbatch run.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl $PERMUTED_ROOT/isomer_splits/test_split.pkl \
#                     --output_dir $PERMUTED_ROOT/isomer_splits/xyz_data/

# sbatch run.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl $PERMUTED_ROOT/rgd1_splits/test_split.pkl \
#                     --output_dir $PERMUTED_ROOT/rgd1_splits/xyz_data/

sbatch run.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl $PERMUTED_ROOT/rgdx_splits/test_split.pkl \
                    --output_dir $PERMUTED_ROOT/rgdx_splits/xyz_data/

# sbatch run.slurm oa2 convert_pkl_to_xyx_permd.py --input_pkl $PERMUTED_ROOT/rgdy_splits/test_split.pkl \
#                     --output_dir $PERMUTED_ROOT/rgdy_splits/xyz_data/