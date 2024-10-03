import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 讀取 .nii 文件
nii_folder_path = '/home/youyu/Medical-SAM2/data/PANCREAS-CT/annotations'

# 使用 glob 列出所有 .nii 文件
nii_files = glob.glob(os.path.join(nii_folder_path, '*.nii'))
print(f'found {len(nii_files)} files.')

# 創建一個總輸出資料夾
output_folder = 'mask'
os.makedirs(output_folder, exist_ok=True)

# 遍歷每個 .nii 檔案
for nii_file in nii_files:
    # 讀取 .nii 文件
    nii_img = nib.load(nii_file)
    img_data = nii_img.get_fdata()

    # 取得文件名稱（不含副檔名）
    file_name = os.path.basename(nii_file).split('.')[0]

    # 為每個文件創建單獨的資料夾來儲存圖片
    file_output_folder = os.path.join(output_folder, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    # 檢查影像資料的維度 (通常是3D的)
    print(f"Processing {file_name}, Image shape: {img_data.shape}")

    # 遍歷每一層 z 儲存為圖片
    idx = 0
    for i in range(0,img_data.shape[2],2):
        # save as .npy files:
        slice_data = img_data[:, :, i]  # 獲取每一層的資料
        # 將這一層儲存為 .npy 文件
        npy_file_path = os.path.join(file_output_folder, f'slice_{idx}.npy')
        np.save(npy_file_path, slice_data)
        idx += 1
        # save as pictures:
        # plt.imshow(img_data[:, :, i], cmap='gray')  # 繪製影像
        # plt.axis('off')  # 隱藏座標軸
        # plt.savefig(os.path.join(file_output_folder, f'slice_{idx}.png'), bbox_inches='tight', pad_inches=0)  # 儲存圖片
        # idx += 1
        # plt.close()

    print(f"Images saved for {file_name} in folder: {file_output_folder}")

print("Processing complete for all .nii files.")


# import numpy as np
# import matplotlib.pyplot as plt

# # 載入 .npy 檔案
# image_data = np.load('/home/youyu/Medical-SAM2/data/btcv/Training/mask/img0001/140.npy')

# # 顯示圖像
# plt.imshow(image_data)  # 如果是灰度圖像，使用 'gray'，若是彩色圖像可以移除 cmap
# plt.show()