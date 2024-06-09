#!/bin/sh

# Split the dataset into multiple img_list files
python3 split.py \
    --img_list "$DATA_DIR/img_list.txt" \
    --partitions "$NUM_PARTITIONS" \
    --output "$DATA_DIR"

# split test dataset into multiple img_list files (if available)
python3 split.py \
    --img_list "data/AFW/img_list.txt" \
    --partitions "$NUM_PARTITIONS" \
    --output "data/AFW/"

python3 split.py \
    --img_list "data/PASCAL/img_list.txt" \
    --partitions "$NUM_PARTITIONS" \
    --output "data/PASCAL/"

python3 split.py \
    --img_list "data/FDDB/img_list.txt" \
    --partitions "$NUM_PARTITIONS" \
    --output "data/FDDB/"

# Read environment variables and pass them to the Python script
python3 client.py \
    --partition_id "$PARTITION_ID" \
    --use_cuda "$USE_CUDA" \
    --server_address "$SERVER_ADDRESS" \
    --data_dir "$DATA_DIR" \
    --img_dim "$IMG_DIM" \
    --num_classes "$NUM_CLASSES" \
    --batch_size "$BATCH_SIZE" \
    --validation_split "$VALIDATION_SPLIT" \
    --test_split "$TEST_SPLIT" \
    --weights_dir "$WEIGHTS_DIR"


tail -f /dev/null