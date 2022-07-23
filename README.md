# Medical Code Assignment via Note-Code Interaction Denoising Network  (NIDN)

## Highlight

- This paper designs a joint learning mechanism to fuse the self-attention matrix and label attention matrix to deal with the long-term dependency problem in clinical documents and extract code-specific text representation.
- Multi-task learning is introduced to obtain code association to assist medical code prediction.
- The proposed model combines the focus loss function and the truncation loss function to design a denoising module. We leverage the focus loss function to give different weights to high-frequency labels and low-frequency labels which can alleviate the class imbalance problem. We use the truncation loss function to discard noisy samples to obtain cleaner ones.


# Package Dependencies

* allennlp == 0.9.0
* ax-platform == 0.1.12
* gensim == 3.8.3
* plotly == 4.7.1
* pytorch==1.5.1
* spacy == 2.1.9
* tensorboardx == 2.0
* tokenizers == 0.7.0
* numpy == 1.15.1
* nltk == 3.5
* python == 3.6.12
* pytorch-pretrained-bert == 0.6.2
* transformers == 2.9.1

You can use the following command (recommended):
~~~
pip install -r requirements.txt
~~~

## Preprossing 

### Clinical Document

We follow the preproces setting of [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network). The structure of data files can be shown like:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions.txt (for DR_CAML)
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
Running the ```python preprocess_mimic3.py``` obtain corresponding ICD code file.

### Obtain CCS dataset

Clinical Classifications Software (CCS) for ICD-9-CM is a tool from HCUP.
Next, download the zip package from [web](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip) and unzip the file. Rename the ```$dxref 2015.csv``` and ```$prref 2015.csv``` as ```dx2015.csv``` and ```pr2015.csv```, respectively. Place two file in the data, the structure is shown like this:

```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions.txt (for DR_CAML)
└───mimic3/
|   |   dev_50_m.csv
|   |   train_50_m.csv
|   |   test_50_m.csv
|   |   dev_full_m.csv
|   |   train_full_m.csv
|   |   test_full_m.csv
|   |   dx2015.csv
|   |   pr2015.csv
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
use the script ```python ICD2CCS.py``` to obtain CCS labels and attach them on corresponding csv files.

## Training

#### MARN
~~~
python main.py --MAX_LENGTH 4000 --bidirectional --n_epochs 50 --batch_size 16 --model GRU --lr 1e-3 --MTL Yes --loss_weight_CCS 0.3 --RAM --task ICD9 --reduce
~~~

## Main Results (all evaluation results are presented in %)
### MIMIC-III-50 (ICD)

| Models     |  Macro AUC-ROC |  Micro AUC-ROC | Macro F1 | Micro F1 |  Precision at 5 |
|--------------|-----------|-----------|-----------|--------------|-----------------------|
|CAML | 91.4 | 93.8 | 62.5 | 68.7 | 65.3 |
|MultiResCNN| 91.7 | 93.9 | 64.1 | 69.0 | 65.0 |
|LAAT| 92.5 | 94.6 | 66.6 | 71.5 | **67.5**|
|JointLAAT| 92.5 | 94.6 | 66.1 | 71.6 | 67.1 |
|[MARN](https://drive.google.com/file/d/1oiiXfw9sn3b21nfqpQSIovxRXLhgMrxZ/view?usp=sharing)    | **92.7** | **94.7** | **68.2** | **71.8** | 67.3 |

### MIMIC-III-full (ICD)

| Models     |  Macro AUC-ROC |  Micro AUC-ROC | Macro F1 | Micro F1 |  Precision at 8  |  Precision at 15  |
|--------------|-----------|-----------|-----------|--------------|-----------------------|-----|
|CAML| 89.5 | 98.6 | 8.8 | 53.9 | 70.9 | 56.1 |
|MultiResCNN| 91.0 | 98.6 | 8.5 | 55.2 | 73.4 | 58.4 |
|LAAT| 91.9 | 98.8 | 9.9 | 57.5 | 73.8 | 59.1 |
|JointLAAT| **92.1** | 98.8 | 10.7 | 57.5 | 73.5 | 59.0 |
|[MARN](https://drive.google.com/file/d/1wwOdb8PDC1N0eOJUP3y4cJ2h6WLZHwjI/view?usp=sharing)    | 91.3 | **98.8** | **11.6** | **58.4** | **75.4** | **60.2** |





## Acknowledgement
We appreciate for all code providers, especially for [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network), [CAML](https://github.com/jamesmullenbach/caml-mimic) and [CCS](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp).
