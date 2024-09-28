from pydicom import dcmread
from pydicom.data import get_testdata_files
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# 取得 Pydicom 附帶的 DICOM 測試影像路徑
filename = get_testdata_files('MR_small.dcm')[0]

# 讀取 DICOM 檔案
ds = dcmread(filename)

# 列出所有後設資料（metadata）
# print(ds)

# 以 matplotlib 繪製影像
plt.imshow(ds.pixel_array)
plt.show()
