##### IMP Index is not the same as the original xyz_files - same index in the .sdf files #####
# Generate TS for Transition1x

# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/transition1x/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/transition1x/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/transition1x/26Mar29_1229PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/transition1x

# Generate TS for isomer

# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/isomer/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/isomer/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/isomer/26Apr05_0507PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/isomer

#Generate TS for rgd1

# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgd1/26Apr05_0507PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgd1

#Generate TS for rgdx

# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdx

# Generate TS for rgdy

# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdy

# # Generate TS for 10% subset
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/transition1x/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/transition1x/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/transition1x_subset_10/26Apr11_0939PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/transition1x_subset_10

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/isomer/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/isomer/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/isomer_subset_10/26Apr11_1038PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/isomer_subset_10

# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgd1_subset_10/26Apr11_1039PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgd1_subset_10

# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx_subset_10/26Apr11_1039PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdx_subset_10

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy_subset_10/26Apr11_1039PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdy_subset_10

# Generate TS for 50% subset
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/transition1x/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/transition1x/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/transition1x_subset_50/26Apr09_1119PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/transition1x_subset_50

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/isomer/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/isomer/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/isomer_subset_50/26Apr09_1119PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/isomer_subset_50

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgd1_subset_50/26Apr09_1149PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgd1_subset_50

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx_subset_50/26Apr10_1229AM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdx_subset_50

# sbatch script_dgx.slurm tsgen predict_new.py \
#     --reactants ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy_subset_50/26Apr11_0647AM/best_model.ckpt \
#     --output_dir ./generated_transition_states/rgdy_subset_50


# Generalisation experiments
# RGD1 -> RGDx
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgd1/26Apr05_0507PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgd1_test_rgdx

# # RGD1 -> RGDy
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgd1/26Apr05_0507PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgd1_test_rgdy

#RGDx -> RGD1
# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdx_test_rgd1

# # RGDx -> RGDy
# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdy/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdy/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdx_test_rgdy

#RGDy -> RGD1
# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgd1/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgd1/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdy_test_rgd1

# #RGDy -> RGDx
# sbatch script_l40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdy_test_rgdx

# RGD1 -> RGDx10
sbatch script_dgx.slurm tsgen predict_new.py \
    --reactants /home/rbs-ml/205030033/ankit/permuted_data/isomer/sdf_files/test_split/reactants/reactants.sdf \
    --products /home/rbs-ml/205030033/ankit/permuted_data/isomer/sdf_files/test_split/products/products.sdf \
    --checkpoint /home/rbs-ml/205030033/ankit/ts_gen/trained_models/isomer/26Apr05_0507PM/best_model.ckpt \
    --output_dir /home/rbs-ml/205030033/ankit/ts_gen/generated_transition_states/isomer_perm

# # RGDx -> RGDx10
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx10/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx10/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdx/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdx_test_rgdx10

# # RGDy -> RGDx10
# sbatch script_a40.slurm tsgen predict_new.py \
#     --reactants ./data/rgdx10/sdf_files/test_split/reactants/reactants.sdf \
#     --products ./data/rgdx10/sdf_files/test_split/products/products.sdf \
#     --checkpoint ./trained_models/rgdy/26Apr05_0525PM/best_model.ckpt \
#     --output_dir ./generated_transition_states/train_rgdy_test_rgdx10