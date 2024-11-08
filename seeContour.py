import cv2
import numpy as np
import os

# 指定包含 .npy 與 .jpg 檔案的資料夾
npy_folder = '/home/Medical-SAM2/data/btcv_btcv/Training/mask/img0005'
jpg_folder = '/home/Medical-SAM2/data/btcv_btcv/Training/image/img0005'
output_folder = '/home/Medical-SAM2/data/btcv_btcv/mixData'

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 讀取資料夾中的所有 .npy 檔案名稱
npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

for npy_file in npy_files:
    # 建立對應的 .jpg 檔案路徑
    npy_path = os.path.join(npy_folder, npy_file)
    jpg_path = os.path.join(jpg_folder, npy_file.replace('.npy', '.jpg'))
    output_path = os.path.join(output_folder, npy_file.replace('.npy', '_overlay.jpg'))
    
    # 確認對應的 .jpg 檔案是否存在
    if not os.path.exists(jpg_path):
        print(f"對應的 JPG 檔案不存在: {jpg_path}")
        continue

    # 讀取 .npy 檔案 (灰階輪廓圖) 和 .jpg 檔案 (背景圖)
    contour_data = np.load(npy_path)
    image = cv2.imread(jpg_path)

    # 確保輪廓和圖片尺寸一致
    contour_data_resized = cv2.resize(contour_data, (image.shape[1], image.shape[0]))

    # 檢查通道數，確保是單通道
    if len(contour_data_resized.shape) > 2:
        contour_data_resized = cv2.cvtColor(contour_data_resized, cv2.COLOR_BGR2GRAY)

    # 將灰階輪廓轉為二值圖像
    _, binary_contour = cv2.threshold(contour_data_resized, 10, 255, cv2.THRESH_BINARY)

    # 找到輪廓
    contours, _ = cv2.findContours(binary_contour.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"找不到任何輪廓: {npy_file}")
        continue

    # 複製圖片並繪製輪廓
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # 紅色輪廓

    # # # 調整透明度
    # # alpha = 0.5
    # # result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 儲存疊加結果
    cv2.imwrite(output_path, overlay)
    print(f"疊加結果已儲存: {output_path}")
