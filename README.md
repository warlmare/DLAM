

# Deep Learning assisted approximate matching (DLAM)

this work was presented as the DFRWS USA 2023: https://www.sciencedirect.com/science/article/pii/S2666281723000793?ref=pdf_download&fr=RR-2&rr=7e510f09ec2e58f0

A more comprehensible google Colab notebook will be coming very soon to play with DLAM. 

---

```
TLDR; you want to verify the results: 

1. check the prerequisites and install all packages
2. create a directory for the test data. Download the Napierone-data as specified below and unzip it.
3. Update the Path_to_original_files: -flag in the testconfig.yml to point at the directory for the test data. 
4. Run: 

python3 evaluation_with_fuzzy_hashes.py evaluation_testcase_FILETYPE/testconfig.yml

with filetype beeing the file type you wish to test, and testconfig.yml beeing the testconfig you just updated. 

```

## Prerequisites:

Ubuntu 21.04 is sufficient to run all modules in this directory. Please run the following command to install prerequisite packages:

`$ sudo apt-get install build-essential libffi-dev python3 python3-dev python3-pip automake autoconf libtool
`

### Installing mrsh-cf and MRSH-v2:

To verify that the two fuzzy hashes `mrsh-cf` and `mrsh-v2` run on your system please run the the following commands: 

`./mrsh -help`

mrsh-cf:

`./mrsh-cf/mrsh_cuckoo.exe -h`

If not please install the necessary libraries that are missing. `libboost-dev` might be needed. Should this not work, all evaluations can be run with **ssdeep** and **TLSH** only. 

### Installing ssdeep and TLSH:

`pip` is required to install all other python packages. 

To install the python3 ssdeep library, run: 

`sudo BUILD_LIB=1 pip install ssdeep`

To install TLSH python3 library, run: 

`pip install python-tlsh`

### Installing all other python libraries:

The following libraries ought to be installed via pip to run the project code: 

```
matplotlib==3.5.1
nilsimsa==0.3.8
nltk==3.6.7
numpy==1.22.1
pandas==1.4.0
plotille==4.0.2
progress==1.6
pycurl==7.45.1
python_tlsh==4.5.0
PyYAML==6.0
simhash==2.1.1
termplotlib==0.3.9
torch==1.11.0
transformers==4.18.0
```
---
### How to aquire the training and test data:

To verify the results only the **test data** -corpus is necessary. It can be downloaded from http://napierone.com.s3.eu-north-1.amazonaws.com/NapierOne/index.html#NapierOne/Data/ . Please download the following file types into a dedicated `./testcorpus` directory:

PDF files:

```
http://napierone.com.s3-eu-north-1.amazonaws.com/NapierOne/Data/PDF/PDF-total.zip
```
XLSX files: 
```
http://napierone.com.s3-eu-north-1.amazonaws.com/NapierOne/Data/XLSX/XLSX-total.zip
```

JAVASCRIPT files: 

```
http://napierone.com.s3-eu-north-1.amazonaws.com/NapierOne/Data/JAVASCRIPT/JAVASCRIPT-total.zip
```
unzip containers. It is possible that the corpus has changed, and that some files are no longer processable via **TLSH** since they are too small or have too little variance. Please remove these files from the corpus and replace them with any other files of the same file type. 

---

Should you whish to train the models please download the training data the following way: 


- govdocs 1: the govdocs corpus can be downloaded via the script: `govdocs_downloader.py` . In line 7 in range() it can be configured how much of the corpus ought to be downloaded (0-700). 
- Training data for XLSX files can be downloaded via the link in this [Paper](https://www.researchgate.net/publication/308861425_Fuse_A_Reproducible_Extendable_Internet-Scale_Corpus_of_Spreadsheets)
- The training data for javascript files can be downloaded [here](https://www.sri.inf.ethz.ch/js150)

In order to remove all files that **TLSH** cannot process please configure directory you wish to screen in `tlsh_test.py` it will remove all files that are not processable by **TLSH**.

---

### How to generate training data:



Use `main.py` to create training data. The following flags and variables have to be set in `main.py`. 

- DATASET_FILETYPE:
    - `all` for mixed files
    - `js` for javascript files
    - `pdf`  for pdf files
    - `xlsx` for xlsx files
    - `random` for random generated files
    - ...
- MAX_FRAGMENT_PERCENTAGE: 
    - in percent from 1 to 99 sets the maximum size that an anomaly can take up in a file. 
- MIN_FRAGMENT_PERCENTAGE: 
    - is the minimum size that an anomaly can take up in training file. 
- SAMPLE_FILES_PATH: 
    - please specify the path to all training data here.
- Line 295, 298, 301, 303:
    - please specifiy where the training datasets ought to be saved. Once for **ssdeep** and once for **TLSH**. Normally they are saved a location that has the following syntax: ``evaluation_testcase_FILETYPE/training_data_for_model``
- Line 293 at the end:
    - specify where the `anomaly` should by saved. This is very important since the anomaly is needed for creating the test data. Normally it is saved at: ``evaluation_testcase_FILETYPE/anomaly``

---
### How to train the models:  

The two scripts ``feedforward.py``  and ``transformer.py`` are used to train models. Specify in line 193 which fuzzy hash is used (either **SSDEEP** or **TLSH**). Specify in line 195 and 196 the paths to the training data. Both the ``normal_...``-data and the ``anomaly_...``-data are needed. In line 199 it can be specified how much of the training should be used for the training. In line 612 it can be configured where the trained model ought to be saved. Normally its a path like ``evaluation_testcase_FILETYPE/model_...``

The easiest way to start the script is to use VISUAL STUDIO CODE. Go to line 605 and with Right-click of the mouse select: 
```Run to line in interactive Window``` .

Than all the loss curve etc. will be plottet etc. 

---
### How to evaluate the models

For every filetype there is a directory like: 

``evluation_testcase_pdf``

In it there is a ``testconfig.yml`` file. This file specifies all the test settings. 

- ``Path_to_original_files:`` is most important. Specify the path where you saved the test data from napierone-corpus after you unpacked them. 
- ``Fragment_size:`` specifies the minimum and maximum size of the anomaly fragment in the test files. 
- ``Fuzzy_hashing_algorithm:`` comment in all fuzzy hashes that you wish to test. If ``mrsh-cf`` and ``MRSH-v2`` should not run on your system leave them commented out. **ssdeep** and **TLSH** should always be used.
- ``Model_Evaluation:`` specifies which model is to be used for evaluation. IMPORTANT: the model has to be named ``Model_Evaluation:`` so you might have to remove the number behind the name like ``_1`` or similiar. Only one model can be used at a time all others have to be commented out. 

To start the evaluation, run: 

```
python3 evaluation_with_fuzzy_hashes.py evaluation_testcase_FILETYPE/testconfig.yml
```

replace filetype with ``PDF, XLSX`` or ``JAVASCRIPT``. The results for the fuzzy hashes are printed out on the commandline. The results of the DLAM model classifications are displayed in the pdf-graphs of the names ```ssdeep_graph_Feedforward.pdf`` and similar. 

If the evaluaion fails. Always make sure to ``rm -rf`` the following directory: 

```
evaluation_testcase_FILETYPE/testfiles
```

otherwise the evaluation cannot run. 

