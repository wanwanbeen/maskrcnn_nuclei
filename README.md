# maskrcnn_nuclei

This repository contains the codes for nuclei instance segmentation using Mask R-CNN.

The code for Mask R-CNN model is adapted from [MatterPort implementation](https://github.com/matterport/Mask_RCNN).

Example data in `data/` is from [Kaggle DSB18](https://www.kaggle.com/c/data-science-bowl-2018) and [Hand-segmented 2D Nuclear Images](http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html).

## Requirements

* TensorFlow 1.4.0
* Keras 2.1.3
* NumPy 
* SciPy
* OpenCV
* scikit-image
* scikit-learn
* Pandas

## How to run

#### Step 1:

* Put your training and test images under data/train and data/test.

#### Step 2: 

(Skip if you do not need mosaic)
* Some small training images may come from the same large image;
* Run nuclei_mosaic.py to recover the original image - this is useful for data augmentation.

```
python nuclei_mosaic.py --TRAIN_DIR data/train --MOSAIC_TRAIN_DIR data/mosaic_train
```

#### Step 3:
* Split training and validation set.

```
python nuclei_trainvalsplit.py
```

#### Step 4:
* Begin training

```
python nuclei_train.py --dir_log logs
```

#### Step 5:
* Inference on validation and test images
* For validation images: also compute mAP
* model_path = the model you want to use; check the name in logs/.

```
python nuclei_inf.py --dir_log logs --model_path logs/nuclei_train20180101T0000/mask_rcnn_nuclei_train_0000.h5
```

#### Step 6:
* Ensemble segmentation results
* model_names = list of models you want to ensemble
* You can set test_flag = False to ensemble validation results instead of test results

```
python nuclei_ensemble.py --test_flag True --model_names nuclei_train20180101T0000_0000 nuclei_train20180102T0000_0001
```

#### Step 7:
* Post-processing and generate run-length encoding

```
python nuclei_postprocess.py
```

## Technical Details and Performance


