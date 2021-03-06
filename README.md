

# Static_Objects_Geolocalization

End-to-end Learning Improves Static Object Geo-localization in Monocular Video:
> [**End-to-end Learning Improves Static Object Geo-localization in Monocular Video**](https://arxiv.org/abs/2004.05232),  
> Mohamed Chaabane, Lionel Gueguen, Ameni Trabelsi, Ross Beveridge, and Stephen O'Hara


    @article{chaabane2020end,
      title={End-to-end Learning Improves Static Object Geo-localization in Monocular Video},
      author={Chaabane, Mohamed and Gueguen, Lionel and Trabelsi, Ameni and Beveridge, Ross and O'Hara, Stephen},
      journal={WACV},
      year={2021}
    }


Contact: [chaabane@colostate.edu](mailto:chaabane@colostate.edu). Any questions or discussion are welcome! 

## Abstract
Accurately estimating the position of static objects, such as traffic lights, from the moving camera of a self-driving car is a challenging problem. In this work, we present a system that improves the localization of static objects by jointly-optimizing the components of the system via learning. Our system is comprised of networks that perform: 1) 5DoF object pose estimation from a single image, 2) association of objects between pairs of frames, and 3) multi-object tracking to produce the final geo-localization of the static objects within the scene. We evaluate our approach using a publicly-available data set, focusing on traffic lights due to data availability. For each component, we compare against contemporary alternatives and show significantly-improved performance. We also show that the end-to-end system performance is further improved via joint-training of the constituent models.

## Installation
* Clone this repo, and run the following commands.
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
git clone https://github.com/MedChaabane/Static_Objects_Geolocalization.git
cd Static_Objects_Geolocalization/
conda create -y -n Objects_Geolocalization python=3.7
conda activate Objects_Geolocalization
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt  # need to pin versions in this file
```

## Extract TLG Data set
In order to extract TLG dataset:
* The nuScenes dataset ( Full dataset and Map expansion )should first downloaded from their [official webpage](https://www.nuscenes.org).
* Install [YOLO v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and download the [pretrained model](https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/weights/download_weights.sh).
```
conda activate Objects_Geolocalization
mkdir YOLO
conda develop YOLO && cd YOLO
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
mv PyTorch-YOLOv3 PyTorchYOLOv3 && cd PyTorchYOLOv3/
pip install -r requirements.txt
cd weights && sh download_weights.sh && cd ..
```

Then, you can change the paths in extract_TLG_dataset.py and run:
```
python extract_TLG_dataset.py
```
## Train Joint Pose and Matching Model

Change the paths and hyperparameters in config/config.py and run:
```
python train_nuscenes.py
```
## Test Tracking and Objects Geo-localization
Download [pretrained model](https://drive.google.com/file/d/1fj60H8sbAstBsiEBYBDWFcqoOsO3tpfY/view?usp=sharing), change path of the model in confog/config.py and run:
```
python test_nuscenes.py
```
## Evaluate Tracking 
In order to evaluate 2D tracking using [py-motmetrics](https://github.com/cheind/py-motmetrics), we provided post-processing code to change ground truth and predictions is same like [MOT Challenge](https://motchallenge.net) format. Change paths in tracking_post_process.py and run:

```
python tracking_post_process.py
```
## Acknowledgement
A large portion of code is borrowed from [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3), [shijieS/SST](https://github.com/shijieS/SST) and [j96w/DenseFusion](https://github.com/j96w/DenseFusion), many thanks to their wonderful work!
