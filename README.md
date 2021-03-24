# IA-NET: Abstraction-based Multi-object Acoustic Anomaly Detection for Low-complexity Big Data Analysis
We propose for the first time a novel one-stage method, called Information-Abstraction-Net (IA-Net), for the detection of abnormal events in multi-object anomaly detection scenarios by utilizing highly abstracted sensory information instead of the entire sampled data set to elevate the transmission and analysis needs of the system.

## Usage     
```
This repo is based on Pytorch 1.6 or higher, please install Pytorch before run.  
Download MIMII dataset and store in the same folder as IA-NET: https://zenodo.org/record/3384388 
pip install -r requirements.txt
```

## Train IA-NET with four sources  
To train IA-Net with four sources (pump, slider, fan and valve), please simply run following code and change the parameters according to your configuration, the default IA-NET is train with batch size 16 and 200 epochs.  
`python train_ids.py --batch_size 16 --epochs 200`  

## Run a simple demo to detect anomaly with four sources  
We provide a simple demo with pre-trained model, the .wav files is stroed in './demo_files', please simply run following code and follow the guide.  
`python demo.py`

## Validate the performence of the IA-NET  
To validate the performance of the IA-NET (**AUC and mAUC**), please run following code  
`python validate_multi.py` 

## Reference  
- [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347

- [2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.
