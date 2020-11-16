import glob2
import math  
import os
import numpy as np
 
files = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
  image_files = glob2.glob(os.path.join("./dataset", ext))
  files += image_files
# Ngẫu nhiên lấy 20% làm Validation 
nb_val = math.floor(len(files)*0.2)
rand_idx = np.random.randint(0, len(files), nb_val)
 
# Tạo file train.txt
with open("train.txt", "w") as f:
  for idx in np.arange(len(files)):
    if (os.path.exists(files[idx][:-3] + "txt")):
      f.write(files[idx]+'\n')
 
# Tạo file vali.txt
with open("val.txt", "w") as f:
  for idx in np.arange(len(files)):
    if (idx in rand_idx) and (os.path.exists(files[idx][:-3] + "txt")):
      f.write(files[idx]+'\n')