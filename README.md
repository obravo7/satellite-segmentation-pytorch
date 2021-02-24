# Satellite Image Segmentation Using PyTorch # 

This repo contains a U-Net implementation for satellite segmentation 


## Data preparation ##

Due to a severe lack of training data, several pre-processing steps are taken to try and 
alleviate this. First, the annotations are converted from their json files to image masks. Then, the satellite images are further tiled
down, to tiles of size `(512x412)`, so that the images can be fed into a fully convolutional network (**FCN**) for semantic segmentation.
The augmentations are: 
  - `blur`: combination of medial and bilateral blur
  - `bright increase`: increase brightness artificially
  - `distort`: elastic deformation of image
  - `gaussian blur`: gaussian blurring
  - `HSV`: convert channels to HSV
  - `medial blur`: median blur
  - `mirror`: mirror image
  - `rotation invariance`: apply rotation invariance (see `data_utils.augment.rotation_invariance()` for details)
  - `crop + resize`: crop image randomly and resize to size expected by network (crop is applied to both image and mask)

### sample of applied augmentations ###

blur |  bright increase  | distort  |  gaussian blur | 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](docs/images/0_0-0_0-blur.png) |  ![](docs/images/0_0-0_0-bright.png)  |  ![](docs/images/0_0-0_0-distort.png) |  ![](docs/images/0_0-0_0-gauss.png) | 


| HSV shift  |  median blur  | mirror  |  gaussian blur + rotation
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](docs/images/0_0-0_0-hsv.png) |  ![](docs/images/0_0-0_0-med-blur.png)  |  ![](docs/images/0_0-0_0-mirror.png) |  ![](docs/images/0_0-0_0-gauss-rot.png)

| distort + rot  |  rotation invariance  | crop + resize  
:-------------------------:|:-------------------------:|:-------------------------:
![](docs/images/0_0-0_0-distort-rt.png) |  ![](docs/images/0_0-0_0-rt-inv.png)  |  ![](docs/images/0_0-0_0-crop-resize.png) 


