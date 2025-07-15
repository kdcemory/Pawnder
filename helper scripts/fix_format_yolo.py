import os
import re

label_dir = "C:\\Users\\kelly\\Desktop\\dog_pose\\val\\labels"  # change this

for fname in os.listdir(label_dir):
    if fname.endswith(".txt"):
        path = os.path.join(label_dir, fname)
        with open(path, "r") as f:
            lines = f.readlines()

        cleaned = []
        for line in lines:
            fixed_line = re.sub(r'\b(\d+)\.0\b', r'\1', line)  # replace 2.0 -> 2
            cleaned.append(fixed_line)

        with open(path, "w") as f:
            f.writelines(cleaned)
