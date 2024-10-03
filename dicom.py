from pydicom import dcmread
from pydicom.data import get_testdata_files
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# 取得 Pydicom 附帶的 DICOM 測試影像路徑
filename = '/home/youyu/Medical-SAM2/data/152435556/00000002' # get_testdata_files('MR_small.dcm')[0]
print(filename)
# 讀取 DICOM 檔案
ds = dcmread(filename)

# 列出所有後設資料（metadata）
# print(ds)

# 以 matplotlib 繪製影像
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.imshow(ds.pixel_array)
plt.savefig('00000005.jpg',bbox_inches='tight',pad_inches=0)
plt.show()

# import os
# import shutil

# # 當前資料夾路徑
# current_folder = os.getcwd()

# # 新資料夾的名稱
# destination_folder = os.path.join(current_folder, "all_files")
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 遍歷當前資料夾內的所有檔案與資料夾
# for root, dirs, files in os.walk(current_folder):
#     # 避免處理我們剛剛創建的新資料夾
#     if root == destination_folder:
#         continue
    
#     for file in files:
#         # 原始檔案路徑
#         file_path = os.path.join(root, file)
        
#         # 複製到新資料夾
#         shutil.move(file_path, destination_folder)

# print(f"所有檔案已移動到 {destination_folder}")
