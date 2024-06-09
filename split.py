import argparse
import random
from pathlib import Path
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_list", required=True, help="Path to the img_list.txt file")
    ap.add_argument("--partitions", required=True, type=int, default=2, help="Number of partitions")
    ap.add_argument("--output", required=True, help="Output directory")
    args = vars(ap.parse_args())

    img_list = args["img_list"]
    partitions = args["partitions"]
    output = args["output"]

    Path(output).mkdir(parents=True, exist_ok=True)

    with open(img_list, "r") as file:
        lines = file.readlines()

    random.shuffle(lines)

    for i in range(partitions):
        with open(os.path.join(output, f"img_list_{i}.txt"), "w") as file:
            for line in lines[i::partitions]:
                file.write(line)
    