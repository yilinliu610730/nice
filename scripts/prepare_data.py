import tarfile
import os
import gzip

data_dir = "data"
listing_file = f"{data_dir}/abo-listings.tar"
listing_dir = f"{data_dir}/listings/metadata"
image_file = f"{data_dir}/abo-images-small.tar"
image_dir = f"{data_dir}/images/metadata"

with tarfile.open(listing_file) as file:
    file.extractall(path=data_dir)

with tarfile.open(image_file) as file:
    file.extractall(path=data_dir)

for meta_dir in [listing_dir, image_dir]:
    for filename in os.listdir(meta_dir):
        if ".gz" in filename:
            filepath = f"{meta_dir}/{filename}"
            filepath_out = filepath.replace(".gz", "")
            with gzip.open(filepath, 'rb') as f:
                file_content = f.read()

            with open(filepath_out, 'wb') as f:
                f.write(file_content)

