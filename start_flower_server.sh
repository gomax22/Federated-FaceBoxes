#!/bin/sh

# Read environment variables and pass them to the Python script
python3 server.py \
    --num_rounds $NUM_ROUNDS \
    --server_address $SERVER_ADDRESS \
    --num_clients $NUM_CLIENTS \
    --img_dim $IMG_DIM \
    --num_classes $NUM_CLASSES \
    --weights_dir $WEIGHTS_DIR

tail -f /dev/null