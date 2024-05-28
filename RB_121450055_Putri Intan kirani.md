Pada artikel Three Ways of Storing and Accessing Lots of Images in Python 
mempelajari berbagai metode penyimpanan dan akses gambar dalam bahasa pemrograman Python
ada tiga metode penyimpanan gambar yaitu :
1. Penyimpanan gambar pada disk sebagai file .png.
2. Penyimpanan gambar dalam basis data berbasis memori-mapping, yaitu LMDB.
3. Penyimpanan gambar dalam format data hierarkis, yaitu HDF5.
setelah itu akan dilakukan membandingkan performa dan penggunaan disk dari ketiga metode tersebut.

1. Mempersiapkan dataset CIFAR-10 akan diunduh dan diproses menggunakan Python
   dataset terdiri dari 60.000 gambar berukuran 32x32 pixel dengan berbagai kelas objek

   import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("data/cifar-10-batches-py/")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")


2. Penyimapan gambar pada disk sebagai file png dengan library Pillow

   from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
3. Penyimpanan gambar berbasis LMDB
   LMDB merupakan basis data berbasis memori-mapping yang cepat dan efisien
   lmdb digunakan untuk mengakses dan menyimpan gambar-gambar dalam LMDB

   import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

  4. Penyimbapan gambar dengan HDF5
     HDF5 adalah format file hierarkis yang cocok untuk menyimpan data ilmiah, termasuk gambar
     h5py digunakan untuk menyimpan gambar-gambar ke dalam file HDF5.

     import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
5. Perbandingan performa penggunaan disk dengan mengukur waktu yang dibutuhkan untuk menyimpan dan membaca gambar dengan setiap metode. Pengukuran waktu
   akan dilakukan menggunakan modul timeit yang disediakan oleh Python.

  import timeit

# Fungsi untuk melakukan pengukuran waktu penyimpanan gambar
def measure_store_time(method, num_images):
    setup_code = f"""
from __main__ import store_single_disk, store_single_lmdb, store_single_hdf5, images, labels, image_ids
num_images = {num_images}
    """
    store_code = """
for i in range(num_images):
    method(images[i], image_ids[i], labels[i])
    """

    time_taken = timeit.timeit(setup=setup_code, stmt=store_code, number=1)
    return time_taken

# Fungsi untuk melakukan pengukuran waktu pembacaan gambar
def measure_read_time(method, num_images):
    setup_code = f"""
from __main__ import read_single_disk, read_single_lmdb, read_single_hdf5, image_ids
num_images = {num_images}
    """
    read_code = """
for i in range(num_images):
    method(image_ids[i])
    """

    time_taken = timeit.timeit(setup=setup_code, stmt=read_code, number=1)
    return time_taken

# Pengukuran waktu penyimpanan gambar
disk_store_time = measure_store_time(store_single_disk, num_images)
lmdb_store_time = measure_store_time(store_single_lmdb, num_images)
hdf5_store_time = measure_store_time(store_single_hdf5, num_images)

# Pengukuran waktu pembacaan gambar
disk_read_time = measure_read_time(read_single_disk, num_images)
lmdb_read_time = measure_read_time(read_single_lmdb, num_images)
hdf5_read_time = measure_read_time(read_single_hdf5, num_images)

import os

# Fungsi untuk menghitung ukuran file pada disk
def get_disk_usage(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Pengukuran penggunaan disk
disk_usage_disk = get_disk_usage(disk_dir)
disk_usage_lmdb = get_disk_usage(lmdb_dir)
disk_usage_hdf5 = get_disk_usage(hdf5_dir)

Dengan melakukan perbandingan performa dan penggunaan disk dari ketiga metode penyimpanan gambar, kita dapat memahami keunggulan dan kelemahan masing-masing metode
Pemilihan metode penyimpanan yang tepat akan sangat ergantung pada kebutuhan aplikasi dan karakteristik dataset yang digunakan.
   
