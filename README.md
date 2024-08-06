# HMM-based Anomaly Detection Algorithm
### Point out the mistakes: An HMM-based Anomaly Detection Algorithm for Sleep Stage Classification
#### by: Ziyi Wang, Hang Liu\*, Yukai Cai, Hongjin Li, Chuanshuai Yang, Xinlei Zhang, Fengyu Cong 


## Requirmenets:
- Python3.7
- Pytorch
- Numpy
- Sklearn
- Pandas

## Prepare datasets
We used two public datasets in this study:
- [Sleep-EDF-20](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b)
- [Sleep-EDF-78](https://physionet.org/content/sleep-edfx/1.0.0/)


After downloading the datasets, you can prepare them as follows:
```
cd prepare_datasets
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
python prepare_shhs.py --data_dir /path/to/EDF/files --ann_dir /path/to/Annotation/files --output_dir shhs_npz --select_ch "EEG C4-A1"
```

We will continue to update the code, usage instructions, and required environment.

## Contact
Hang Liu   
School of Biomedical Engineering, Faculty of Medicine,

Dalian University of Technology, Dalian, China   
Email: liuahng@dlut.edu.cn
