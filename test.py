import os

folder_path = '/home/youyu/Medical-SAM2/Training/images'

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # 檢查文件名中是否包含 "PANCREASE"
        if "slice" in filename:
            # 新文件名：將 "PANCREASE" 替換為 "img"
            new_filename = filename.replace("slice_", "")
            
            # 舊文件的完整路徑
            old_file_path = os.path.join(root, filename)
            # 新文件的完整路徑
            new_file_path = os.path.join(root, new_filename)
            
            # 重新命名文件
            os.rename(old_file_path, new_file_path)
            
            print(f'Renamed: {old_file_path} -> {new_file_path}')

print("All matching files in the folder and its subfolders have been renamed.")