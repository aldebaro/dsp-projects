# Spatio-Temporal Filter Adaptive Network for Video Deblurring

Video deblurring is a challenging task due to the spatially variant blur caused by camera shake, object motions, and depth variations, etc. Existing methods usually estimate optical flow in the blurry video to align consecutive frames or approximate blur kernels. However, they tend to generate artifacts or cannot effectively remove blur when the estimated optical flow is not accurate. To overcome the limitation of separate optical flow estimation, we propose a Spatio-Temporal Filter Adaptive Network (STFAN) for the alignment and deblurring in a unified framework. The proposed STFAN takes both blurry and restored images of the previous frame as well as blurry image of the current frame as input, and dynamically generates the spatially adaptive filters for the alignment and deblurring. We then propose the new Filter Adaptive Convolutional (FAC) layer to align the deblurred features of the previous frame with the current frame and remove the spatially variant blur from the features of the current frame. Finally, we develop a reconstruction network which takes the fusion of two transformed features to restore the clear frames. Both quantitative and qualitative evaluation results on the benchmark datasets and real-world videos demonstrate that the proposed algorithm performs favorably against state-of-the-art methods in terms of accuracy, speed as well as model size. 

# Basic information about the project

Main paper / reference: Zhou, S., Zhang, J., Pan, J., Xie, H., Zuo, W., & Ren, J. (2019). Spatio-temporal filter adaptive network for video deblurring. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 2482-2491).

Main dataset: https://paperswithcode.com/paper/spatio-temporal-filter-adaptive-network-for

Original code: https://github.com/sczhou/STFAN

Language: Python3

# Installation

Instructions for installing

# Executing / performing basic analysis

Provide information on how to execute the main code, how to obtain results, etc. Provide the name of the main scripts.

# Credits

date - your name - your github URL

# References

The main references you used
