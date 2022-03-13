

## Main Results



### Partial Labeled Data

We followed STAC[1] to evaluate on 5 different data splits for each setting, and report the average performance of 5 splits. The results are shown in the following:

## PublayNet [3]

#### 1% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  82.5 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)   | 84.9|[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|
| Student-Teacher (Faster R-CNN ResNet101)| 87.4 |[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py)|

#### 5% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  83.3 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)   | 87.2|[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|
| Student-Teacher (Faster R-CNN ResNet101)| 86.7 |[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py)|

#### 10% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  83.4 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)  | 87.3|[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|
| Student-Teacher (Faster R-CNN ResNet101)| 88.9 |[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py)|

#### Fully labeled baseline comparision
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Faster R-CNN [3]|  90.2 |  |
| Student-Teacher (Faster R-CNN ResNet101 + 10% labeled data)| 90.0 |[Config](configs/soft_teacher_publaynet_fasterrcnn/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py)|


### IIIT-AR-13K [4]

#### 1% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  35.8 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)   | 42.2 |[Config](configs/soft_teacher_iiitar_faster_rcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|


#### 5% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  49.7 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)   | 51.8 |[Config](configs/soft_teacher_iiitar_faster_rcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|


#### 10% labeled data
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| Supervised |  57.4 |[Config](configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py)|
| Student-Teacher (Faster R-CNN ResNet50)  | 63.3|[Config](configs/soft_teacher_iiitar_faster_rcnn/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py)|


#### Fully labeled baseline comparision
| Method | mAP| Config Files|
| ---- | -------| ----- |----|
| YOLO F [5]|  90.2 |  |
| Student-Teacher (Faster R-CNN ResNet50 + 10% labeled data)| 63.3 |[Config](configs/soft_teacher_iiitar_faster_rcnn/soft_teacher_faster_rcnn_r101_caffe_fpn_coco_full_1080k.py)|



### Notes
- Ours* means we use longer training schedule.
- `thr` indicates `model.test_cfg.rcnn.score_thr` in config files. This inference trick was first introduced by Instant-Teaching[2].
- All models are trained on 8*V100 GPUs

## Usage

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.9.0`
- `mmdetection=2.16.0+fe46ffe`
- `mmcv=1.3.9`
- `wandb=0.10.31`

#### Notes
- We use [wandb](https://wandb.ai/) for visualization, if you don't want to use it, just comment line `273-284` in `configs/soft_teacher/base.py`.
- The project should be compatible to the latest version of `mmdetection`. If you want to switch to the same version `mmdetection` as ours, run `cd thirdparty/mmdetection && git checkout v2.16.0`
### Installation
```
make install
```

### Data Preparation
- Download the COCO dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/
#     unlabeled2017/
#     annotations/
ln -s ${YOUR_DATA} data
bash tools/dataset/prepare_coco_data.sh conduct

```
For concrete instructions of what should be downloaded, please refer to `tools/dataset/prepare_coco_data.sh` line [`11-24`](https://github.com/microsoft/SoftTeacher/blob/863d90a3aa98615be3d156e7d305a22c2a5075f5/tools/dataset/prepare_coco_data.sh#L11)
### Training
- To train model on the **partial labeled data** setting:
```shell script
# JOB_TYPE: 'baseline' or 'semi', decide which kind of job to run
# PERCENT_LABELED_DATA: 1, 5, 10. The ratio of labeled coco data in whole training dataset.
# GPU_NUM: number of gpus to run the job
for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh <JOB_TYPE> ${FOLD} <PERCENT_LABELED_DATA> <GPU_NUM>
done
```
For example, we could run the following scripts to train our model on 10% labeled data with 8 GPUs:

```shell script
for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh semi ${FOLD} 10 8
done
```

- To train model on the **full labeled data** setting:

```shell script
bash tools/dist_train.sh <CONFIG_FILE_PATH> <NUM_GPUS>
```
For example, to train ours `R50` model with 8 GPUs:
```shell script
bash tools/dist_train.sh configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py 8
```
- To train model on **new dataset**:

The core idea is to convert a new dataset to coco format. Details about it can be found in the [adding new dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_dataset.md).



### Evaluation
```
bash tools/dist_test.sh <CONFIG_FILE_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval bbox --cfg-options model.test_cfg.rcnn.score_thr=<THR>
```
### Inference
  To inference with trained model and visualize the detection results:

  ```shell script
  # [IMAGE_FILE_PATH]: the path of your image file in local file system
  # [CONFIG_FILE]: the path of a confile file
  # [CHECKPOINT_PATH]: the path of a trained model related to provided confilg file.
  # [OUTPUT_PATH]: the directory to save detection result
  python demo/image_demo.py [IMAGE_FILE_PATH] [CONFIG_FILE] [CHECKPOINT_PATH] --output [OUTPUT_PATH]
  ```
  For example:
  - Inference on single image with provided `R50` model:
   ```shell script
  python demo/image_demo.py /tmp/tmp.png configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py work_dirs/downloaded.model --output work_dirs/
  ```

  After the program completes, a image with the same name as input will be saved to `work_dirs`

  - Inference on many images with provided `R50` model:
   ```shell script
  python demo/image_demo.py '/tmp/*.jpg' configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py work_dirs/downloaded.model --output work_dirs/
  ```

[1] [A Simple Semi-Supervised Learning Framework for Object Detection](https://arxiv.org/pdf/2005.04757.pdf)

[2] [Instant-Teaching: An End-to-End Semi-SupervisedObject Detection Framework](https://arxiv.org/pdf/2103.11402.pdf)

[3] [PubLayNet: largest dataset ever for document layout 406 analysis] (https://arxiv.org/abs/1908.07836)

[4] [IIIT-AR-13K: A New Dataset for Graphical Object Detection in Documents] (https://arxiv.org/abs/2008.02569)

[5] [Page Object Detection with YOLOF] (https://ieeexplore.ieee.org/document/9701449)
