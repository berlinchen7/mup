python main.py --optimizer adam --lr 0.01 \
                --cuda --coord_check \
                --coord_check_nsteps 3 \
                --nlayers 4 \
                --bptt 35 \
                --hyperparam_mode mup_fullalign --d_model_base 3 \
                --coord_check_nseeds 5

# Default bptt 35 coord_check_nsteps 3 coord_check_nseeds 3