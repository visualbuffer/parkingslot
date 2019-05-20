# PARKING SLOT DETECTION
![Detection Parking Slots](assets/carslots.png)
Detecting parking slots occupancy from surveilance cam video feed

# DOWNLOAD THE IMAGES
! wget http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar
! tar -xvf  CNR-EXT_FULL_IMAGE_1000x750.tar
! rm CNR-EXT_FULL_IMAGE_1000x750.tar

! wget http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
! unzip CNR-EXT-Patches-150x150.zip
! rm CNR-EXT-Patches-150x150.zip

! wget http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
! unzip  CNR-EXT-Patches-150x150.zip
! rm CNR-EXT-Patches-150x150.zip


# DOWNLOAD THE MODEL

YOLO

! wget https://pjreddie.com/media/files/yolov3.weights
! python ./convert.py ./yolov3.cfg yolov3.weights model_data/yolo.h5


MASKRCNN
! wget "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

