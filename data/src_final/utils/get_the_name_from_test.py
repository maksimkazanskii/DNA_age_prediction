import pandas as pd
import os
import shutil
test_split_file = "experiments/experiment_article_new/SPLIT_DATA/test.csv"
full_bam_path = "data/raw_full/damageProfiler"
target_bam_path = "data/only_test_HARVARD"
target_folder = "data/only_test_HARVARD"
df = pd.read_csv(test_split_file)
os.makedirs(target_bam_path,exist_ok=True)
for name in df['bam_name']:
    path_src = os.path.join(full_bam_path, name+".bam")
    path_tgt = os.path.join(target_folder, name+".bam")
    print("Copying ", path_src)
    shutil.copytree(path_src, path_tgt)
