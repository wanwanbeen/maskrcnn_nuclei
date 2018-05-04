# maskrcnn_nuclei

This repository contains the codes for nuclei instance segmentation using Mask R-CNN.
The codes for Mask R-CNN are adapted from [MatterPort implementation](https://github.com/matterport/Mask_RCNN).
Data used to train and validate is from [Kaggle DSB18](https://www.kaggle.com/c/data-science-bowl-2018).

## Files

* ```nuclei_model.py```: model for Mask R-CNN 
* ```nuclei_config.py```: configuration for Mask R-CNN 
* ```nuclei_utils.py```: utility functions 
* ```nuclei_mosaic.py```: mosaic images for data augmentation
* ```nuclei_main_train.py```: train the models
* ```nuclei_main_inf.py```: inference on test images
* ```nuclei_ensemble.py```: test time ensemble
* ```nulcei_postprocess.m```: post-processing
* ```nuclei_postprocess_mask2rle.m```: convert segmentation mask to run length encoding

## Requirements

* TensorFlow
* Keras 
* NumPy
* SciPy
* OpenCV
* scikit-image
* scikit-learn
* Pandas

