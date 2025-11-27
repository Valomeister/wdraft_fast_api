# import torch
# import static_data
#
# checkpoint = torch.load("models/icon_randomcrop_model.pth", map_location="cpu")
# # checkpoint = torch.load("models/character_cnn.pth", map_location="cpu")
#
# state_dict = checkpoint["model"]      # веса модели
# classes = checkpoint["classes"]       # список классов
#
# print(classes)
#
# new_classes = ['ALLI', 'AMBER', 'ANGELO', 'ASH', 'BARLEY', 'BEA', 'BELLE', 'BERRY', 'BIBI', '8-BIT', 'BO', 'BONNIE', 'BROCK', 'BULL', 'BUSTER', 'BUZZ', 'BYRON', 'CARL', 'CHARLIE', 'CHESTER', 'CHUCK', 'CLANCY', 'COLETTE', 'COLT', 'CORDELIUS', 'CROW', 'DARRYL', 'DOUG', 'DRACO', 'DYNAMIKE', 'EDGAR', 'EMZ', 'EVE', 'FANG', 'FINX', 'FRANK', 'GALE', 'GENE', 'GRAY', 'GRIFF', 'GROM', 'GUS', 'HANK', 'JACKY', 'JAE-YONG', 'JANET', 'JESSIE', 'JUJU', 'KAZE', 'KENJI', 'KIT', 'LARRY & LAWRIE', 'LEON', 'LILY', 'LOLA', 'LOU', 'LUMI', 'MAISIE', 'MANDY', 'MAX', 'MEEPLE', 'MEG', 'MELODIE', 'MICO', 'MINA', 'MOE', 'MORTIS', 'MR. P', 'NANI', 'NITA', 'OLLIE', 'OTIS', 'PAM', 'PEARL', 'PENNY', 'PIPER', 'POCO', 'EL PRIMO', 'RICO', 'ROSA', 'R-T', 'RUFFS', 'SAM', 'SANDY', 'SHADE', 'SHELLY', 'SPIKE', 'SPROUT', 'SQUEAK', 'STU', 'SURGE', 'TARA', 'TICK', 'TRUNK', 'WILLOW', 'ZIGGY']
#
# # mapping = {i: (static_data.BRAWLERS.index(x) if x in static_data.BRAWLERS else -1)for i, x in enumerate(new_classes)}
# #
# # print(mapping)
#
# for i in range(len(classes)):
#     print(classes[i], new_classes[i])
#
# # torch.save({ "model": state_dict, "classes": new_classes }, "models/icon_randomcrop_model.pth")




import os
import shutil

src_dir = "picks/training_set_png_crop"
dst_dir = "picks/training_set_png_crop_"

new_classes = ['ALLI', 'AMBER', 'ANGELO', 'ASH', 'BARLEY', 'BEA', 'BELLE', 'BERRY', 'BIBI', '8-BIT', 'BO', 'BONNIE', 'BROCK', 'BULL', 'BUSTER', 'BUZZ', 'BYRON', 'CARL', 'CHARLIE', 'CHESTER', 'CHUCK', 'CLANCY', 'COLETTE', 'COLT', 'CORDELIUS', 'CROW', 'DARRYL', 'DOUG', 'DRACO', 'DYNAMIKE', 'EDGAR', 'EMZ', 'EVE', 'FANG', 'FINX', 'FRANK', 'GALE', 'GENE', 'GRAY', 'GRIFF', 'GROM', 'GUS', 'HANK', 'JACKY', 'JAE-YONG', 'JANET', 'JESSIE', 'JUJU', 'KAZE', 'KENJI', 'KIT', 'LARRY & LAWRIE', 'LEON', 'LILY', 'LOLA', 'LOU', 'LUMI', 'MAISIE', 'MANDY', 'MAX', 'MEEPLE', 'MEG', 'MELODIE', 'MICO', 'MINA', 'MOE', 'MORTIS', 'MR. P', 'NANI', 'NITA', 'OLLIE', 'OTIS', 'PAM', 'PEARL', 'PENNY', 'PIPER', 'POCO', 'EL PRIMO', 'RICO', 'ROSA', 'R-T', 'RUFFS', 'SAM', 'SANDY', 'SHADE', 'SHELLY', 'SPIKE', 'SPROUT', 'SQUEAK', 'STU', 'SURGE', 'TARA', 'TICK', 'TRUNK', 'WILLOW', 'ZIGGY']

# создаём папку назначения, если её нет
os.makedirs(dst_dir, exist_ok=True)

files = sorted(os.listdir(src_dir))  # берём файлы в стабильном порядке

for old_name, new_name in zip(files, new_classes):
    new_name = new_name + ".png"
    src_path = os.path.join(src_dir, old_name)
    dst_path = os.path.join(dst_dir, new_name)
    print(os.path.basename(src_path), os.path.basename(dst_path))
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)  # можно заменить на shutil.move
