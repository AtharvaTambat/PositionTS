# Set these to your local paths before running.
DATA_ROOT="${DATA_ROOT:-../data}"                 # dataset splits (shared with the OA-ReactDiff experiments)
PERMUTED_ROOT="${PERMUTED_ROOT:-../data/permuted}"  # atom-order-permuted copies of the splits
MODEL_ROOT="${MODEL_ROOT:-./trained_models}"      # trained checkpoints
OUTPUT_ROOT="${OUTPUT_ROOT:-./generated}"         # generated transition states

# TS1x dataset
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/transition1x/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/transition1x/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/transition1x/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/transition1x/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/transition1x/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/transition1x/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/transition1x/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/transition1x/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/transition1x/sdf_files/test_split/products/products.sdf \

# Isomers
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/isomer/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/isomer/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/isomer/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/isomer/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/isomer/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/isomer/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/isomer/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/isomer/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/isomer/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/isomer/

# RGD1
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/rgd1/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/rgd1/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/rgd1/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/rgd1/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/rgd1/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/rgd1/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/rgd1/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/rgd1/

# RGDx
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/rgdx/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/rgdx/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/rgdx/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/rgdx/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/rgdx/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/rgdx/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/rgdx/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/rgdx/

# RGDy
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/rgdy/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/rgdy/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/rgdy/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/rgdy/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/rgdy/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/rgdy/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/rgdy/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/rgdy/ \
#     --restore $MODEL_ROOT/rgdy/26Apr05_0525PM/last_model.ckpt

# Training for 50% train samples
# sbatch run.slurm tsgen train.py \
#     --train_r ./data/transition1x_subset_50/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/transition1x_subset_50/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/transition1x_subset_50/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/transition1x_subset_50/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/transition1x_subset_50/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/transition1x_subset_50/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/transition1x_subset_50/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/transition1x_subset_50/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/transition1x_subset_50/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/transition1x_subset_50/


# sbatch run.slurm tsgen train.py \
#     --train_r ./data/isomer_subset_50/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/isomer_subset_50/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/isomer_subset_50/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/isomer_subset_50/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/isomer_subset_50/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/isomer_subset_50/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/isomer_subset_50/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/isomer_subset_50/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/isomer_subset_50/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/isomer_subset_50/

# sbatch run.slurm tsgen train.py \
#     --train_r ./data/rgd1_subset_50/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/rgd1_subset_50/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/rgd1_subset_50/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/rgd1_subset_50/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/rgd1_subset_50/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/rgd1_subset_50/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/rgd1_subset_50/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/rgd1_subset_50/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/rgd1_subset_50/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/rgd1_subset_50/

# sbatch run.slurm tsgen train.py \
#     --train_r ./data/rgdx_subset_50/sdf_files/train_split/reactants/reactants.sdf \
#     --train_ts ./data/rgdx_subset_50/sdf_files/train_split/transition_states/transition_states.sdf \
#     --train_p ./data/rgdx_subset_50/sdf_files/train_split/products/products.sdf \
#     --valid_r ./data/rgdx_subset_50/sdf_files/validation_split/reactants/reactants.sdf \
#     --valid_ts ./data/rgdx_subset_50/sdf_files/validation_split/transition_states/transition_states.sdf \
#     --valid_p ./data/rgdx_subset_50/sdf_files/validation_split/products/products.sdf \
#     --test_r ./data/rgdx_subset_50/sdf_files/test_split/reactants/reactants.sdf \
#     --test_ts ./data/rgdx_subset_50/sdf_files/test_split/transition_states/transition_states.sdf \
#     --test_p ./data/rgdx_subset_50/sdf_files/test_split/products/products.sdf \
#     --save_checkpoint_dir ./trained_models/rgdx_subset_50/

sbatch run.slurm tsgen train.py \
    --train_r ./data/rgdy_subset_50/sdf_files/train_split/reactants/reactants.sdf \
    --train_ts ./data/rgdy_subset_50/sdf_files/train_split/transition_states/transition_states.sdf \
    --train_p ./data/rgdy_subset_50/sdf_files/train_split/products/products.sdf \
    --valid_r ./data/rgdy_subset_50/sdf_files/validation_split/reactants/reactants.sdf \
    --valid_ts ./data/rgdy_subset_50/sdf_files/validation_split/transition_states/transition_states.sdf \
    --valid_p ./data/rgdy_subset_50/sdf_files/validation_split/products/products.sdf \
    --test_r ./data/rgdy_subset_50/sdf_files/test_split/reactants/reactants.sdf \
    --test_ts ./data/rgdy_subset_50/sdf_files/test_split/transition_states/transition_states.sdf \
    --test_p ./data/rgdy_subset_50/sdf_files/test_split/products/products.sdf \
    --save_checkpoint_dir ./trained_models/rgdy_subset_50/