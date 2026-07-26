# sbatch run.slurm tsgan train.py --data_path ./data/transition1x_subset_50_numpy/train/train.py.npy \
#                     --trained_model_path ./generated_transition_states/transition1x_subset_50

# sbatch run.slurm tsgan train.py --data_path ./data/isomer_subset_50_numpy/train/train.npy \
#                     --trained_model_path ./generated_transition_states/isomer_subset_50

# sbatch run.slurm tsgan train.py --data_path ./data/rgd1_subset_50_numpy/train/train.npy \
#                     --trained_model_path ./generated_transition_states/rgd1_subset_50

# sbatch run.slurm tsgan train.py --data_path ./data/rgdx_subset_50_numpy/train/train.npy \
#                     --trained_model_path ./generated_transition_states/rgdx_subset_50

# sbatch run.slurm tsgan train.py --data_path ./data/rgdy_subset_50_numpy/train/train.npy \
#                     --trained_model_path ./generated_transition_states/rgdy_subset_50

sbatch run.slurm tsgan train.py --data_path ./data/rgd1_numpy/train/train.npy \
                    --trained_model_path ./generated_transition_states/rgd1