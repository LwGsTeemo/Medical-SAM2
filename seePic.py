import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import mclahe as mc
# 讀取 .nii 文件
nii_folder_path = '/home//Medical-SAM2/data/PANCREAS-CT/images'
npy_folder_path = '/home//Medical-SAM2/data/btcv/Test/mask/img0001/'
test_folder_path = '/home//Medical-SAM2/data/test/image/115.jpg'

def rotatePicture(test_folder_path):
    from PIL import Image
    # 讀取圖片
    image = Image.open(test_folder_path)
    # 旋轉圖片90度 (逆時針方向)
    rotated_image = image.rotate(0, expand=True)
    # 顯示旋轉後的圖片
    rotated_image.show()
    # 保存旋轉後的圖片
    rotated_image.save('./data/test/rotated.jpg')

def seeNpyPic(npy_folder_path):
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    # np.set_printoptions(threshold=np.inf)
    for i in range(240):
        data = np.load(f'{npy_folder_path}/{i}.npy')
        if np.max(data) != 0:
            print(i)
            plt.imshow(data)
            plt.savefig(f'./picture/{i}.jpg')
            plt.show()
            # image = Image.fromarray(data)
            # if image.mode == 'F':
            #     image = image.convert('RGB')
            # image.save(f'./picture/{i}.jpg')

def readNii(nii_folder_path):
    # 使用 glob 列出所有 .nii 文件
    nii_files = glob.glob(os.path.join(nii_folder_path, '*.nii'))
    print(f'found {len(nii_files)} files.')

    # 創建一個總輸出資料夾
    output_folder = 'image'
    os.makedirs(output_folder, exist_ok=True)

    # 遍歷每個 .nii 檔案
    for nii_file in nii_files:
        # 讀取 .nii 文件
        nii_img = nib.load(nii_file)
        img_data = nii_img.get_fdata()

        # 前處理
        img_data = np.clip(img_data, -300,500)
        img_data = img_data.astype(np.uint8)
        img_data = mc.mclahe(img_data, kernel_size=(8, 8, 8),
                      n_bins=128,
                      clip_limit=0.0,
                      adaptive_hist_range=False,)
        img_data = 255-((img_data*255.).astype(np.uint8).clip(0, 255))
        mi, ma = np.min(img_data), np.max(img_data)
        img_data = (img_data - mi) / (ma - mi)
        img_data = img_data.astype("float32")

        # 取得文件名稱（不含副檔名）
        file_name = os.path.basename(nii_file).split('.')[0]

        # 為每個文件創建單獨的資料夾來儲存圖片
        file_output_folder = os.path.join(output_folder, file_name)
        os.makedirs(file_output_folder, exist_ok=True)

        # 檢查影像資料的維度 (通常是3D的)
        print(f"Processing {file_name}, Image shape: {img_data.shape}")

        # 遍歷每一層 z 儲存為圖片
        idx = 0
        for i in range(img_data.shape[2]):
            # # save as .npy files:
            # slice_data = img_data[:, :, i]  # 獲取每一層的資料
            # # 將這一層儲存為 .npy 文件
            # npy_file_path = os.path.join(file_output_folder, f'{idx}.npy')
            # np.save(npy_file_path, slice_data)
            # idx += 1
            # save as pictures:
            plt.imshow(img_data[:, :, i], cmap='gray')  # 繪製影像
            plt.axis('off')  # 隱藏座標軸
            plt.savefig(os.path.join(file_output_folder, f'{i}.jpg'), bbox_inches='tight', pad_inches=0)  # 儲存圖片
            idx += 1
            plt.close()

        print(f"Images saved for {file_name} in folder: {file_output_folder}")

    print("Processing complete for all .nii files.")

def savePic():
    import os
    import glob
    # 設定主資料夾路徑，內含多個子資料夾
    main_folder_path = '/home/Medical-SAM2/data/btcv_btcv/Training/mask'
    # 設定計數器來追踪新檔名
    counter = 0

    # 遍歷主資料夾中的每個子資料夾
    for subfolder in sorted(os.listdir(main_folder_path)):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        
        # 確認該子資料夾路徑是否存在且為資料夾
        if os.path.isdir(subfolder_path):
            # 找出子資料夾內的所有 jpg 圖片，按名稱順序排序
            images = sorted(glob.glob(os.path.join(subfolder_path, '*.npy')))
            
            # 遍歷每張圖片，依次重新命名
            for image_path in images:
                new_name = f"{counter}.npy"
                new_path = os.path.join(main_folder_path, new_name)
                os.rename(image_path, new_path)
                counter += 1

    print("圖片重新命名完成！")

def transferPic():
    import os
    import numpy as np
    from PIL import Image
    import mclahe as mc
    import cv2
    # 設定要讀取圖片的資料夾路徑
    folder_path = '/home/Medical-SAM2/data/btcv_btcv/Training/image'

    # 遍歷資料夾內的所有 .jpg 檔案
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(folder_path, file_name)
            
            # 讀取圖片並轉為 numpy 陣列
            img = cv2.imread(file_path)

            # 依次進行處理
            # convert the image into grayscale before doing histogram equalization
            # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_data = mc.mclahe(img[:, :, 0])
            
            # 進行反轉及範圍裁剪
            img_data = 255 - ((img_data * 255.).astype(np.uint8).clip(0, 255))
            
            # 正規化處理
            mi, ma = np.min(img_data), np.max(img_data)
            img_data = (img_data - mi) / (ma - mi)
            img_data = img_data.astype("float32")
            
            # 將處理後的圖片存回新檔案
            final_path = '/home/Medical-SAM2/data/btcv_btcv/Training/afterImage'
            output_path = os.path.join(final_path, f'{file_name}')
            Image.fromarray((img_data * 255).astype(np.uint8)).save(output_path)

    print("圖片處理完成！")


if __name__ == '__main__':
    readNii(nii_folder_path)
    # seeNpyPic(npy_folder_path)
    # rotatePicture(test_folder_path)
    # savePic()
    # transferPic()