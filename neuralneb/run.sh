sbatch run.slurm oa2 scripts/neb.py ../ts_gen/data/transition1x/xyz_files/test_split/products/entry_0.xyz ../ts_gen/data/transition1x/xyz_files/test_split/reactants/entry_0.xyz

# sbatch run.slurm oa2 scripts/neb_new.py \
#   --reactant_dir ../ts_gen/data/transition1x/xyz_files/test_split/reactants/ \
#   --product_dir ../ts_gen/data/transition1x/xyz_files/test_split/products/ \
#   --output_dir ./generated_transition_states/transition1x \
#   --model data/painn.sd