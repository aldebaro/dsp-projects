# Spatio-Temporal Filter Adaptive Network for Video Deblurring

Video deblurring is a challenging task due to the spatially variant blur caused by camera shake, object motions, and depth variations, etc. Existing methods usually estimate optical flow in the blurry video to align consecutive frames or approximate blur kernels. However, they tend to generate artifacts or cannot effectively remove blur when the estimated optical flow is not accurate. To overcome the limitation of separate optical flow estimation, we propose a Spatio-Temporal Filter Adaptive Network (STFAN) for the alignment and deblurring in a unified framework. The proposed STFAN takes both blurry and restored images of the previous frame as well as blurry image of the current frame as input, and dynamically generates the spatially adaptive filters for the alignment and deblurring. We then propose the new Filter Adaptive Convolutional (FAC) layer to align the deblurred features of the previous frame with the current frame and remove the spatially variant blur from the features of the current frame. Finally, we develop a reconstruction network which takes the fusion of two transformed features to restore the clear frames. Both quantitative and qualitative evaluation results on the benchmark datasets and real-world videos demonstrate that the proposed algorithm performs favorably against state-of-the-art methods in terms of accuracy, speed as well as model size. 

# Basic information about the project

Main paper / reference: Zhou, S., Zhang, J., Pan, J., Xie, H., Zuo, W., & Ren, J. (2019). Spatio-temporal filter adaptive network for video deblurring. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 2482-2491).

Main dataset: S. Su, M. Delbracio, J. Wang, G. Sapiro, W. Heidrich, and
O. Wang. Deep video deblurring for hand-held cameras. In
CVPR, 2017

Original code: https://github.com/sczhou/STFAN

Presentation: https://docs.google.com/presentation/d/1Q2avz-cBu1ZRE-a1_5_q5r9vadDUllwLzdyUWvbtq0I/edit?usp=sharing

Language: Python2

# Installation

Instructions for installing

CUDA 8.0/9.0/10.0
gcc 4.9+
Python 2.7
PyTorch 1.0+
easydict
enum34
matplotlib
scipy
opencv-python==4.2.0.32

```bash
git clone https://github.com/sczhou/STFAN.git
```

Crete the virtuaenv with python2
```bash
mkvirtualenv -p python2.7 pds
```

```bash
pip install matplotlib easydict scipy
pip install opencv-python==4.2.0.32
pip install torch===1.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install enum34 unique typing
```

```bash
pip install git+https://github.com/jamesbowman/openexrpython.git
pip install pyexr
pip install torch===1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX=='2.0'
```

### To-do
add informations about code modification

```bash
python runner.py  --phase 'test'  --weights './ckpt/best-ckpt.pth.tar'   --data './dataset_root' --out './output'
```


# Executing / performing basic analysis

Provide information on how to execute the main code, how to obtain results, etc. Provide the name of the main scripts.

# Credits

date - your name - your github URL

# References

The main references you used
