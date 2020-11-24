

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
* Clone this repo, and we'll call the directory that you cloned as ${OBJECTS_GEOLOCALIZATION}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0
```
conda create -n Objects_Geolocalization
conda activate Objects_Geolocalization
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${OBJECTS_GEOLOCALIZATION}
pip install -r requirements.txt
```

## Extract TLG Data set
In order to extract TLG dataset:
* The nuScenes dataset should first downloaded from their [official webpage](https://www.nuscenes.org).
* Install [YOLO v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and download the [pretrained model](https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/weights/download_weights.sh).
