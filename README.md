[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# IA-Net: Abstraction-based Multi-object Acoustic Anomaly Detection for Low-complexity Big Data Analysis
We propose for the first time a novel one-stage method, called Information-Abstraction-Net (IA-Net), for the detection of abnormal events in multi-object anomaly detection scenarios by utilizing highly abstracted sensory information instead of the entire sampled data set to elevate the transmission and analysis needs of the system.

## Table of Contents
- [IA-Net: Abstraction-based Multi-object Acoustic Anomaly Detection for Low-complexity Big Data Analysis](#ia-net-abstraction-based-multi-object-acoustic-anomaly-detection-for-low-complexity-big-data-analysis)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Train IA-Net with four sources](#train-ia-net-with-four-sources)
  - [Run demo](#run-demo)
  - [Validate the performence](#validate-the-performence)
  - [Citation](#citation)
  - [About Us](#about-us)
  - [License](#license)
  - [Reference](#reference)

## Usage     
This repo is based on Pytorch 1.6 or higher, please install Pytorch.

Download [MIMII](https://zenodo.org/record/3384388) dataset [1, 2] and store in the same folder as IA-Net: 
```
pip install -r requirements.txt
```

## Train IA-Net with four sources  
To train IA-Net with four sources (pump, slider, fan and valve), please simply run following code and change the parameters according to your configuration, the default IA-NET is train with batch size 16 and 200 epochs:
```
python train_ids.py --batch_size 16 --epochs 200
```

## Run demo  
We provide a simple demo with pre-trained model, the .wav files is stroed in './demo_files', please simply run following code and follow the guide:
```
python demo.py
```

## Validate the performence 
To validate the performance of the IA-Net (**AUC and mAUC**), please run following code:
```
python validate_multi.py
```

## Citation

If you like our repository, please cite our papers.

``` 
@INPROCEEDINGS{Wu2106:Abstraction,
AUTHOR={Huanzhuo Wu and Jia He and M{\'a}t{\'e} {T{\"o}m{\"o}sk{\"o}zi} and Frank H.P. Fitzek},
TITLE="Abstraction-based Multi-object Acoustic Anomaly Detection for Low-complexity Big Data Analysis",
BOOKTITLE="WS17 IEEE ICC 2021 Workshop on Communication, Computing, and Networking in Cyber-Physical Systems (WS17 ICC'21 Workshop - CCN-CPS)",
ADDRESS="Montreal, Canada",
DAYS=14,
MONTH=jun,
YEAR=2021,
KEYWORDS="anomaly detection; big data; IoT; industry 4.0",
ABSTRACT="In the deployments of cyber-physical systems, specifically predictive maintenance and Internet of Things applications, a staggering amount of data can be harvested, transmitted, and recorded. Although the collection of large data sets can be used for many solutions, its utilization is made difficult by the increased overhead on the transmission and limited processing capabilities of the underlying physical system. For such highly correlated and extensive data, this situation is usually described as data-rich, information-poor. We propose for the first time a novel one-stage method, called Information-Abstraction-Net (IA-Net), for the detection of abnormal events in multi-object anomaly detection scenarios by
utilizing highly abstracted sensory information instead of the entire sampled data set to elevate the transmission and analysis needs of the system. We find that the computation complexity of IA-Net is reduced by half compared to competing solutions and the detection accuracy is increased by about 5-47\%, as well."
}
```
## About Us

We are researchers at the Deutsche Telekom Chair of Communication Networks (ComNets) at TU Dresden, Germany. Our focus is on in-network computing.

* **Huanzhuo Wu** - huanzhuo.wu@tu-dresden.de
* **Jia He** - jia.he@mailbox.tu-dresden.de

## License

This project is licensed under the [MIT license](./LICENSE).

## Reference  
- [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347

- [2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.
