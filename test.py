import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("GPU count:", torch.cuda.device_count())
#
# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))


# import os
#
# root_dir = r"D:\JetBrains\PycharmProjects\reserch"
# max_level = 2  # 1=一级，2=二级
#
# for root, dirs, files in os.walk(root_dir):
#     level = root.replace(root_dir, "").count(os.sep) + 1
#     if level > max_level:
#         continue
#     indent = " " * 4 * (level - 1)
#     print(f"{indent}{os.path.basename(root)}/")
#     for f in files:
#         print(f"{indent}    {f}")

import datasets
print(datasets.__version__)
print(hasattr(datasets, "Dataset"))  # True