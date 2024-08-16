# srun --nodes=1 \
#     --ntasks-per-node=2 \
#     --gres=gpu:2 \
#     --time=01:00:00 \
#     --pty bash -i

srun --nodes=1 \
    --nodelist=mk-ix-07 \
    --time=7-01:00:00 \
    --pty bash -i



# srun --nodes=1 \
#     --mem=516G  \
#     --nodelist=mk-ix-07 \
#     --gres=gpu:1 \
#     --time=7-05:00:00 \
#     --pty bash -i \