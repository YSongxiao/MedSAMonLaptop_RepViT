import nrrd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from tqdm import tqdm


k_path = "Raw_Data/KiTS/"
d_path = "Raw_Data/Dongyang/"
r_path = "Raw_Data/Rider/"
save_path = "Other-Datasets/h5/CT/Seg-A/"
k_img_list = sorted([str(path) for path in Path(k_path).rglob("[A-Za-z][0-9].nrrd") if path.is_file()])
k_img_list += sorted([str(path) for path in Path(k_path).rglob("[A-Za-z][0-9][0-9].nrrd") if path.is_file()])
d_img_list = sorted([str(path) for path in Path(d_path).rglob("[A-Za-z][0-9].nrrd") if path.is_file()])
d_img_list += sorted([str(path) for path in Path(d_path).rglob("[A-Za-z][0-9][0-9].nrrd") if path.is_file()])
r_img_list = sorted([str(path) for path in Path(r_path).rglob("[A-Za-z][0-9].nrrd") if path.is_file()])
r_img_list += sorted([str(path) for path in Path(r_path).rglob("[A-Za-z][0-9][0-9].nrrd") if path.is_file()])

k_seg_list = sorted([str(path) for path in Path(k_path).rglob("*.seg.nrrd") if path.is_file()])
d_seg_list = sorted([str(path) for path in Path(d_path).rglob("*.seg.nrrd") if path.is_file()])
r_seg_list = sorted([str(path) for path in Path(r_path).rglob("*.seg.nrrd") if path.is_file()])

for file in tqdm(k_img_list):
    seg_file = file.replace(".nrrd", ".seg.nrrd")
    img_data, img_options = nrrd.read(file)
    seg_data, seg_options = nrrd.read(seg_file)
    img_data = img_data.transpose(2, 0, 1)
    seg_data = seg_data.transpose(2, 0, 1)
    h5filename = Path(file).stem + ".h5"
    h5filepath = save_path + h5filename
    with h5py.File(h5filepath, 'w') as hdf:
        hdf.create_dataset('imgs', data=img_data)
        hdf.create_dataset('gts', data=seg_data)

for file in tqdm(d_img_list):
    seg_file = file.replace(".nrrd", ".seg.nrrd")
    img_data, img_options = nrrd.read(file)
    seg_data, seg_options = nrrd.read(seg_file)
    img_data = img_data.transpose(2, 0, 1)
    seg_data = seg_data.transpose(2, 0, 1)
    h5filename = Path(file).stem + ".h5"
    h5filepath = save_path + h5filename
    with h5py.File(h5filepath, 'w') as hdf:
        hdf.create_dataset('imgs', data=img_data)
        hdf.create_dataset('gts', data=seg_data)

for file in tqdm(r_img_list):
    seg_file = file.replace(".nrrd", ".seg.nrrd")
    img_data, img_options = nrrd.read(file)
    seg_data, seg_options = nrrd.read(seg_file)
    img_data = img_data.transpose(2, 0, 1)
    seg_data = seg_data.transpose(2, 0, 1)
    h5filename = Path(file).stem + ".h5"
    h5filepath = save_path + h5filename
    with h5py.File(h5filepath, 'w') as hdf:
        hdf.create_dataset('imgs', data=img_data)
        hdf.create_dataset('gts', data=seg_data)

# img_path = "/mnt/data2/datasx/MedSAM/Other-Datasets/Raw_Data/KiTS/K1/K1.nrrd"
# seg_path = "/mnt/data2/datasx/MedSAM/Other-Datasets/Raw_Data/KiTS/K1/K1.seg.nrrd"
# img_data, img_options = nrrd.read(img_path)
# # data_np = np.array(img_data)
# seg_data, seg_options = nrrd.read(seg_path)
# # seg_np = np.array(seg_data)
# plt.imshow(img_data[..., 60], 'gray')
# plt.show()
# plt.imshow(seg_data[..., 60], 'gray')
# plt.show()
