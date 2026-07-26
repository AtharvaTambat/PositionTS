# Set these to your local paths before running.
DATA_ROOT="${DATA_ROOT:-../../data}"                 # dataset splits (shared with the OA-ReactDiff experiments)
PERMUTED_ROOT="${PERMUTED_ROOT:-../../data/permuted}"  # atom-order-permuted copies of the splits
MODEL_ROOT="${MODEL_ROOT:-./trained_models}"      # trained checkpoints
OUTPUT_ROOT="${OUTPUT_ROOT:-./generated}"         # generated transition states

# Generate TS For TS1x test data
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/transition1x/xyz_files/test_split/reactants \
#     ../ts_gen/data/transition1x/xyz_files/test_split/products \
#     ./trained_models/ts1x/g_model.h5 \
#     ./generated_transition_states/transition1x

# Generate TS For isomer test data
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/isomer/xyz_files/test_split/reactants \
#     ../ts_gen/data/isomer/xyz_files/test_split/products \
#     ./trained_models/isomer/g_model.h5 \
#     ./generated_transition_states/isomer

# Generate TS For rgdx test data
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx/xyz_files/test_split/products \
#     ./trained_models/rgdx/g_model.h5 \
#     ./generated_transition_states/rgdx

# Generate TS For rgdy test data
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdy/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdy/xyz_files/test_split/products \
#     ./trained_models/rgdy/g_model.h5 \
#     ./generated_transition_states/rgdy

# Subset 50 generate xyz molecules
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/transition1x/xyz_files/test_split/reactants \
#     ../ts_gen/data/transition1x/xyz_files/test_split/products \
#     ./trained_models/transition1x_subset_50/g_model.h5 \
#     ./generated_transition_states/transition1x_subset_50

# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/isomer/xyz_files/test_split/reactants \
#     ../ts_gen/data/isomer/xyz_files/test_split/products \
#     ./trained_models/isomer_subset_50/g_model.h5 \
#     ./generated_transition_states/isomer_subset_50

# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgd1/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgd1/xyz_files/test_split/products \
#     ./trained_models/rgd1_subset_50/g_model.h5 \
#     ./generated_transition_states/rgd1_subset_50

# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx/xyz_files/test_split/products \
#     ./trained_models/rgdx_subset_50/g_model.h5 \
#     ./generated_transition_states/rgdx_subset_50

# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdy/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdy/xyz_files/test_split/products \
#     ./trained_models/rgdy_subset_50/g_model.h5 \
#     ./generated_transition_states/rgdy_subset_50

# Generalisation experiments
# RGD1 -> RGDx 
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx/xyz_files/test_split/products \
#     ./trained_models/rgd1/g_model.h5 \
#     ./generated_transition_states/train_rgd1_test_rgdx

# # RGD1 -> RGDy
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdy/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdy/xyz_files/test_split/products \
#     ./trained_models/rgd1/g_model.h5 \
#     ./generated_transition_states/train_rgd1_test_rgdy

# # RGDx -> RGD1
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgd1/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgd1/xyz_files/test_split/products \
#     ./trained_models/rgdx/g_model.h5 \
#     ./generated_transition_states/train_rgdx_test_rgd1

# RGDx -> RGDy
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdy/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdy/xyz_files/test_split/products \
#     ./trained_models/rgdx/g_model.h5 \
#     ./generated_transition_states/train_rgdx_test_rgdy

# RGDy -> RGD1
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgd1/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgd1/xyz_files/test_split/products \
#     ./trained_models/rgdy/g_model.h5 \
#     ./generated_transition_states/train_rgdy_test_rgd1

# RGDy -> RGDx
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx/xyz_files/test_split/products \
#     ./trained_models/rgdy/g_model.h5 \
#     ./generated_transition_states/train_rgdy_test_rgdx

# RGDy -> RGDx10
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx10/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx10/xyz_files/test_split/products \
#     ./trained_models/rgdy/g_model.h5 \
#     ./generated_transition_states/train_rgdy_test_rgdx10

# # RGDx -> RGDx10
# sbatch run.slurm tsgan predict.py \
#     ../ts_gen/data/rgdx10/xyz_files/test_split/reactants \
#     ../ts_gen/data/rgdx10/xyz_files/test_split/products \
#     ./trained_models/rgdx/g_model.h5 \
#     ./generated_transition_states/train_rgdx_test_rgdx10

# Generate TS For rgd1 test data
sbatch run.slurm tsgan predict.py \
    ../ts_gen/data/rgd1/xyz_files/test_split/reactants \
    ../ts_gen/data/rgd1/xyz_files/test_split/products \
    ./trained_models/rgd1/g_model.h5 \
    ./generated_transition_states/rgd1

# RGD1 -> RGDx
sbatch run.slurm tsgan predict.py \
    ../ts_gen/data/rgdx/xyz_files/test_split/reactants \
    ../ts_gen/data/rgdx/xyz_files/test_split/products \
    ./trained_models/rgd1/g_model.h5 \
    ./generated_transition_states/train_rgd1_test_rgdx

# RGD1 -> RGDy
sbatch run.slurm tsgan predict.py \
    ../ts_gen/data/rgdy/xyz_files/test_split/reactants \
    ../ts_gen/data/rgdy/xyz_files/test_split/products \
    ./trained_models/rgd1/g_model.h5 \
    ./generated_transition_states/train_rgd1_test_rgdy

# RGD1 -> RGDx10
sbatch run.slurm tsgan predict.py \
    ../ts_gen/data/rgdx10/xyz_files/test_split/reactants \
    ../ts_gen/data/rgdx10/xyz_files/test_split/products \
    ./trained_models/rgd1/g_model.h5 \
    ./generated_transition_states/train_rgd1_test_rgdx10

# Generate TS For permuted rgd1 test data
sbatch run.slurm tsgan predict.py \
    $PERMUTED_ROOT/rgd1_splits/xyz_data/reactant \
    $PERMUTED_ROOT/rgd1_splits/xyz_data/product \
    ./trained_models/rgd1/g_model.h5 \
    ./generated_transition_states/rgd1_permd