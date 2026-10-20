# Learning-Dual-Camera-Smooth-Zoom-via-Decoupled-Geometric-Photometric-Data-Synthesis
Code of Learning Dual-Camera Smooth Zoom via Decoupled Geometric-Photometric Data Synthesis

## 1.Abstract
<p align="center"><img src="./assets/intro.png" width="95%"></p>
<!-- When zooming between dual cameras on smartphones, prominent discontinuities in both geometric perspective and color rendition inevitably degrade the visual experience of users. To address this, we introduce the dual-camera smooth zoom (DCSZ) task, aiming to synthesize continuous intermediate frames for fluid zooming. However, directly applying existing video frame interpolation (VFI) models to this task is severely hindered by substantial domain gaps and the physical intractability of capturing real-world continuous ground truth. To this end, we propose a novel data synthesis framework to generate high-fidelity DCSZ training sequences, named ZoomGS v2. It explicitly decouples geometric trajectories and photometric transformations from 3D representation modeling, thereby enabling extensive data generation from easily accessible single-camera 3D scenes. Furthermore, we design ZoomFI, a specialized VFI network tailored to the DCSZ task, which leverages comple- mentary bidirectional optical flows for comprehensive dual-camera information fusion. Extensive experiments on real-world datasets of two mobile phones demonstrate that training with our constructed DCSZ data significantly improves the performance of various VFI models. Moreover, the proposed ZoomFI achieves state-of-the-art results in both quantitative metrics and visual quality. -->
When zooming between dual cameras on smartphones, prominent discontinuities in both geometric perspective and color 
rendition inevitably degrade the visual experience of users. To address this, we introduce the dual-camera smooth 
zoom (DCSZ) task, aiming to synthesize continuous intermediate frames for fluid zooming.  However, directly applying 
existing video frame interpolation (VFI) models to this task is severely hindered by substantial domain gaps and the 
physical intractability of capturing real-world continuous ground-truth. To this end, we propose a novel data synthesis 
framework to generate high-fidelity DCSZ training sequences, named ZoomGS v2. It explicitly decouples geometric 
trajectories and color transformations from 3D representation modeling, thereby enabling extensive data generation 
from easily accessible single-camera 3D scenes. Furthermore, we design ZoomFI, a specialized VFI network tailored 
to the DCSZ task, which leverages complementary bidirectional warping for comprehensive dual-camera information fusion. 
Extensive experiments on test datasets of two mobile phones demonstrate that training with our constructed DCSZ data 
significantly improves the performance of various VFI models. Moreover, the proposed ZoomFI achieves state-of-the-art 
results in both quantitative metrics and visual quality.

## 2.Method
### 2.1 Comparison of ZoomGS and ZoomGS v2
<p align="center"><img src="./assets/pipline_synzoomgs.png" width="95%"></p>
(a) Pipline of ZoomGS. ZoomGS models the scene with a Base Gaussians and use CamTrans Module to learn spatial and photometric transitions between the two endpoint 3DGS, and then transform the Base Gaussians to target Gaussians of virtual cameras for rendering.<br />
(b) Pipline of ZoomGS v2. ZoomGS v2 first calibrate distribution of camera and color parameters from small dual-camera data, and then applies them to Gaussians constructed from large single-camera data for 3D scene rendering. The producted example on the right illustrates that ZoomGS v2 generates images with better quality than ZoomGS.

### 2.2 Zoom FI
<p align="center"><img src="./assets/FI_model.png" width="55%"></p>
Structure of ZoomFI.

## 3.Prerequisites and Datasets
### 3.1 Prerequisites
- Python 3.8.16, PyTorch 2.1.1, **cuda-11.8**
<!-- - We provide detailed dependencies in [`environment.yml`] for Real-ZoomGS and ZoomFI, and [`./SynZoomGS/environment.yml`] for Syn-ZoomGS. -->

### 3.2 Datasets

### 3.3 Pretrained models

## 4.Start ZoomGS v2
Coming soon.
<!-- - Run [`cd ./SynZoomGS`](./SynZoomGS)
- For xiaomi data, zoom factor of W set to 0.6, 
  Run [`bash ./zoomgs_render_xiaomi06.sh`](./zoomgs_trains.sh)
- For xiaomi data, zoom factor of W set to 0.85, 
  Run [`bash ./zoomgs_render_xiaomi85.sh`](./zoomgs_trains.sh)
- For huawei data, zoom factor of W set to 0.6, 
  Run [`bash ./zoomgs_render_huawei06.sh`](./zoomgs_trains.sh)
- For huawei data, zoom factor of W set to 0.85, 
  Run [`bash ./zoomgs_render_huawei85.sh`](./zoomgs_trains.sh) -->

## 5.Start for ZoomFI
- Run [`cd ./ZoomFI`](./)
- Training: run [`bash ./train.sh`](./train.sh)
- Inference test data: run [`bash ./inference.sh`](./inference.sh)
- Image quality testing: 
    - Run [`cd ./test_img`] (./test_img)
    - Run [`bash ./test.sh`](./test.sh)
- Video quality testing: 
    - Run [`cd ./test_video`] (./test_video)
    - Run [`cd ./DOVER-master`] (./test_video/DOVER-master)
    - Run [`bash ./test.sh`](./test.sh)
    - Run [`cd ../FAST-VQA-and-FasterVQA-dev`] (./test_video/FAST-VQA-and-FasterVQA-dev)
    - Run [`bash ./test.sh`](./test.sh)
- Colr continuity testing:
    - Run [`cd ./test_color`] (./test_color)
    - Run [`bash ./crop_img.sh`](./crop_img.sh)
    - Run [`bash ./test_color.sh`](./test_color.sh)

## Acknowledgement

Special thanks to the following awesome projects!

- [DCSZ](https://github.com/ZcsrenlongZ/ZoomGS)
- [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- [DL3DV-10K](https://github.com/DL3DV-10K/Dataset)
- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [BAD-Gaussian](https://github.com/WU-CVGL/BAD-Gaussians)
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE)

## Citation
If you make use of our work, please cite our paper.
