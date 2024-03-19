# VSCNet
PyTorch's implementation of "A Virtual-Sensor Construction Network Based on Physical Imaging for Image Super-Resolution".

# Requirements
+ Python 3.8
+ PyTorch 1.8.1 (>1.1.0)
+ cuda 11.3

# Preparing Datasets
Download following datasets:
> DIV2K, Flickr2K, Set5, Set14, B100, Urban100: Datas can be download from the official website. 

> RawSR: 

Place the datasets in the ./DataSet folder, and the detailed path for placing the data can be found in Our_dataloader_xxx.py

# Trainning
+ RAW

```
python Train_raw_x2.py --experiment_indxe '<Results save path>'
```

+ RGB
```
python Train_rgb_x4.py --experiment_indxe '<Results save path>'
```



