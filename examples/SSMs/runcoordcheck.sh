python main.py --optimizer adam --lr 0.1 \
                --cuda --coord_check \
                --coord_check_nsteps 10 \
                --nlayers 1 \
                --bptt 1 \
                --hyperparam_mode sp_fullalign --d_model_base 3 \
                --coord_check_nseeds 30 \
                --no_warning #--coord_check_norm rms \

# Default bptt 35 coord_check_nsteps 3 coord_check_nseeds 3 d_model_base 6
# steps 10 bptt 35
