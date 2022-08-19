import pycurl 
from zipfile import ZipFile
import shutil
import os
import glob

for i in range(0,500):
    CTR = str(i).zfill(3)
    print("start downloading {}.zip".format(CTR))
    LINK = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/{}.zip".format(CTR)
    FILENAME = "{}.zip".format(CTR)
    with open(FILENAME, "wb") as f: 
        cl = pycurl.Curl()
        cl.setopt(cl.URL, LINK)
        cl.setopt(cl.WRITEDATA, f)
        cl.perform()
        cl.close()
    print("finished downloading {}.zip".format(CTR))
    print("start unzipping {}.zip".format(CTR))
    with ZipFile("{}.zip".format(CTR), "r") as zipObj:
        zipObj.extractall("./all_files")
    print("finished unzipping {}.zip".format(CTR))
    os.remove("{}.zip".format(CTR))
    file_names = os.listdir("./all_files/{}".format(CTR))
    for file_name in file_names:
        #shutil.copy(os.path.join("all_files/{}".format(CTR),"/", file_name), "all_files")
        shutil.move(os.path.join("all_files/{}".format(CTR), file_name), "all_files")
    print("moved all files to all_files/")
    os.rmdir("all_files/{}".format(CTR))
    tifCounter = len(glob.glob1("all_files/", "*.pdf"))
    print(tifCounter, " PDF files")
    print("-" * 100)
