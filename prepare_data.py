import pandas as pd
import os
import shutil

# chemins
csv_path = "bee_data.csv"
images_path = "bee_imgs/bee_imgs/"   #  CORRIGÉ
output_path = "dataset/"

# lire CSV
df = pd.read_csv(csv_path)

# créer dossier dataset
if not os.path.exists(output_path):
    os.makedirs(output_path)

# compteur pour debug
count = 0

# parcourir les données
for index, row in df.iterrows():
    img_name = row['file']
    label = row['health'].replace(" ", "_")

    class_dir = os.path.join(output_path, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    src = os.path.join(images_path, img_name)
    dst = os.path.join(class_dir, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
        count += 1
    else:
        print("Image non trouvée :", src)

print(f"Organisation terminée | {count} images copiées")