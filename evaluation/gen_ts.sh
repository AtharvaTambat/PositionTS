# Set these to your local paths before running.
DATA_ROOT="${DATA_ROOT:-../baselines/ts_gen/data}"            # per-dataset xyz_files/test_split trees
INFERENCE_ROOT="${INFERENCE_ROOT:-../baselines/learnTS}"      # inference .pkl files and generated output

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/transition1x/xyz_files/test_split/reactants $DATA_ROOT/transition1x/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_ts1x10_ep823_test_ts1x100.pkl $INFERENCE_ROOT/generated_ts/transition1x_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/isomer/xyz_files/test_split/reactants $DATA_ROOT/isomer/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_isomer10_ep535_test_isomer100.pkl $INFERENCE_ROOT/generated_ts/isomer_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgd1/xyz_files/test_split/reactants $DATA_ROOT/rgd1/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_rgd1_10_ep597_test_rgd1_100.pkl $INFERENCE_ROOT/generated_ts/rgd1_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx/xyz_files/test_split/reactants $DATA_ROOT/rgdx/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_rgdx10_ep661_test_rgdx100.pkl $INFERENCE_ROOT/generated_ts/rgdx_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdy/xyz_files/test_split/reactants $DATA_ROOT/rgdy/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_rgdy10_ep565_test_rgdy100.pkl $INFERENCE_ROOT/generated_ts/rgdy_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgd1/xyz_files/test_split/reactants $DATA_ROOT/rgd1/xyz_files/test_split/products $INFERENCE_ROOT/learnability/s10/train_rgd1_10_ep597_test_rgd1_100.pkl $INFERENCE_ROOT/generated_ts/rgd1_s10

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx/xyz_files/test_split/reactants $DATA_ROOT/rgdx/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgd1_ep956_test_rgdx.pkl $INFERENCE_ROOT/generated_ts/trained_rgd1_test_rgdx/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdy/xyz_files/test_split/reactants $DATA_ROOT/rgdy/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgd1_ep956_test_rgdy.pkl $INFERENCE_ROOT/generated_ts/trained_rgd1_test_rgdy/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx10/xyz_files/test_split/reactants $DATA_ROOT/rgdx10/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgd1_ep956_test_rgdx10.pkl $INFERENCE_ROOT/generated_ts/trained_rgd1_test_rgdx10/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgd1/xyz_files/test_split/reactants $DATA_ROOT/rgd1/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdx_ep368_test_rgd1.pkl $INFERENCE_ROOT/generated_ts/trained_rgdx_test_rgd1/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdy/xyz_files/test_split/reactants $DATA_ROOT/rgdy/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdx_ep368_test_rgdy.pkl $INFERENCE_ROOT/generated_ts/trained_rgdx_test_rgdy/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx10/xyz_files/test_split/reactants $DATA_ROOT/rgdx10/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdx_ep368_test_rgdx10.pkl $INFERENCE_ROOT/generated_ts/trained_rgdx_test_rgdx10/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgd1/xyz_files/test_split/reactants $DATA_ROOT/rgd1/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdy_ep249_test_rgd1.pkl $INFERENCE_ROOT/generated_ts/trained_rgdy_test_rgd1/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx/xyz_files/test_split/reactants $DATA_ROOT/rgdx/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdy_ep249_test_rgdx.pkl $INFERENCE_ROOT/generated_ts/trained_rgdy_test_rgdx/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx10/xyz_files/test_split/reactants $DATA_ROOT/rgdx10/xyz_files/test_split/products $INFERENCE_ROOT/generalization_inference_files/train_rgdy_ep249_test_rgdx10.pkl $INFERENCE_ROOT/generated_ts/trained_rgdy_test_rgdx10/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/transition1x/xyz_files/test_split/reactants $DATA_ROOT/transition1x/xyz_files/test_split/products $INFERENCE_ROOT/permutation_inference_files/train_ts1x_ep369_test_ts1x_permute.pkl $INFERENCE_ROOT/generated_ts/transition1x_permd/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/isomer/xyz_files/test_split/reactants $DATA_ROOT/isomer/xyz_files/test_split/products $INFERENCE_ROOT/permutation_inference_files/train_isomer_ep75_test_isomer_permuted.pkl $INFERENCE_ROOT/generated_ts/isomer_permd/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgd1/xyz_files/test_split/reactants $DATA_ROOT/rgd1/xyz_files/test_split/products $INFERENCE_ROOT/permutation_inference_files/train_rgd1_ep956_test_rgd1_permuted.pkl $INFERENCE_ROOT/generated_ts/rgd1_permd/

sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdx/xyz_files/test_split/reactants $DATA_ROOT/rgdx/xyz_files/test_split/products $INFERENCE_ROOT/permutation_inference_files/train_rgdx_ep368_test_rgdx_permute.pkl $INFERENCE_ROOT/generated_ts/rgdx_permd/

# sbatch run.slurm oa2 generate_ts.py $DATA_ROOT/rgdy/xyz_files/test_split/reactants $DATA_ROOT/rgdy/xyz_files/test_split/products $INFERENCE_ROOT/permutation_inference_files/train_rgdy_ep249_test_rgdy_permute.pkl $INFERENCE_ROOT/generated_ts/rgdy_permd/
