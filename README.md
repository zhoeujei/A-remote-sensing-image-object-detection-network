# A-remote-sensing-image-object-detection-network
Note: Some of the code comes from https://github.com/bubbliiiing/yolox-pytorch.


#The innovation part has shown effective improvements on the DIOR dataset, and a paper is currently under review. 
#If it is accepted, we will share the link to the paper in the future.
#Description of Innovations:

The work of this paper mainly consists
of four parts, namely: Additional Predictive Feature Layer module (APFL), Fusion Standard
Deviation Feature Enhancement Module (FSEM), Adaptive Feature Fusion Module (AFM), and
an improved loss function. 

APFL: The proposed idea of APFL is as follows: YOLOX-Tiny cannot effectively cover all
objects, especially dense and proportionally different remote sensing objects. For a remote sensing
image with an input resolution of 416×416, the model uses only three scales of feature maps for
prediction, which are 13×13, 26×26, and 52×52, and can effectively recognize objects larger than
32 pixels × 32 pixels, 16 pixels × 16 pixels, and 8×8 pixels. However, for objects smaller than 8
pixels × 8 pixels, the detection loss probability is relatively high. Therefore, we added the APFL
module to improve the detection accuracy of small targets. The Improved algorithm ablation
experiments in section 18 of the paper show that APFL improved the detection accuracy of targets
by 0.52%. 

FSEM: Attention mechanisms can make it easier for networks to focus on targets in the
background. However, most attention mechanisms currently used, such as SENet, CBAM, and
ECANet, use global average pooling or global maximum pooling to replace image features as the
learning prerequisite for attention coefficients, ignoring the prerequisite of global average pooling
or global maximum pooling as the learning prerequisite for attention coefficients, resulting in poor
performance of attention mechanisms in complex background remote sensing images. Based on
this, we designed the FSEM module, which integrates the standard deviation values and the
attention coefficients of the entire dataset to effectively focus on the targets in remote sensing
images containing noise information. The effectiveness of the FSEM module was verified in the
Improved algorithm ablation experiments section on page 18 of the paper. 

AFM:The well-known feature pyramid network FPN and its variants fuse shallow and deep
features. However, this fusion method is the same and static for the entire dataset, ignoring the
feature fusion considering the specific input image, which undoubtedly limits the adaptiveness of
feature fusion and affects the fusion effect. To address this issue, we propose the AFM module, combining the idea of residual network,
which can perform multi-scale feature adaptive fusion
according to the different features of specific input images. The effectiveness of the AFM module
is validated in the Improved algorithm ablation experiments section on page 18 of the paper. 

Loss function: To improve the misalignment problem between the decoupled regression and
classification tasks in the YOLOX-Tiny algorithm, this paper modified the original algorithm's
loss function. The modified loss function fuses the predicted confidence values and the predicted
classification results, enabling the classification and regression tasks to interact more effectively
during network training. The effectiveness of the modified loss function was verified in the
Improved algorithm ablation experiments section on page 18 of the paper
