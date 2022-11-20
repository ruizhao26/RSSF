# Real Scenes with Spike and Flow (RSSF)

This repository contains the download link and deployment code for our dataset:  
Real Scenes with Spike and Flow (RSSF),  
which is proposed in  
[**Learninng Optical Flow from Continuous Spike Streams**](https://github.com/ruizhao26/Spike2Flow)  
**NeurIPS 2022**

## Introduction to RSSF

Real scenes with spike and flow (RSSF) is a dataset for training and evaluating spike-based optical flow. The dataset is generated based on data in [Slow Flow dataset](http://www.cvlibs.net/projects/slow_flow/) that is captured by high-speed cameras. There are 31 scenes for training and 10 scenes for testing. There are three kinds of training scenes with different resolutions and numbers of spike frames. There are a total of 9.6k+ flow fields and 193k+ spike frames in the training dataset. As for the 11 scenes in the evaluation dataset, we select the first 200 flow fields in each scene. To standardize the evaluation data, we use center clipped to make the width of each spike frame and flow field to be 1024 and make the height of images whose height exceeds 768 to be 768. The totals of flow fields and spike frames are 2.2k and 44.22k, respectively. Noted that the ``Number of Flow Fields'' only counts the flow in $dt=20$ case. The number of flow fields in $dt=40$ and $dt=60$ cases is similar to that in the $dt=20$ case.

Statistics of the RSSF

<img src="https://github.com/ruizhao26/RSSF/blob/main/figs/rssf_statistics.png" alt="rssf_statistics" style="zoom: 33%;" />

Scenes of training part of the RSSF:

<img src="https://github.com/ruizhao26/RSSF/blob/main/figs/train_scenes.png" alt="train_scenes" style="zoom:33%;" />

Scenes of the testing part of the RSSF:

<img src="https://github.com/ruizhao26/RSSF/blob/main/figs/test_scenes.png" alt="test_scenes" style="zoom:33%;" />

## Download the dataset



## Deploy the Dataset

As described in the paper, the flow is generated from the images, and the spikes are generated based on the images and flow. For saving the storage, we only release the spikes and images of the RSSF since the flow fields need much more storage compared with the spikes and images. The flow can be generated locally based on the images and the codes.

```bash
cd generate_flow &&
# please set your rssf root in the follow .py file
python3 generate_flow.py --interval 20 &&
python3 generate_flow.py --interval 40 &&
python3 generate_flow.py --interval 60
```

## Cication

If you find this code useful in your research, please consider citing our paper.

```
@inproceedings{zhao2022learning,
  title={Learninng optical flow from continuous spike streams},
  author={Zhao, Rui and Xiong, Ruiqin and Zhao, Jing and Yu, Zhaofei and Fan, Xiaopeng and Huang, Tiejun},
  booktitle={Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

If you have any questions, please contact:  
ruizhao@stu.pku.edu.cn

## Acknowledgement

Parts of this code were derived from [GMA](https://github.com/zacjiang/GMA), please also consider to cite [Slow Flow](https://www.cvlibs.net/projects/slow_flow/) if you'd like to cite our dataset.