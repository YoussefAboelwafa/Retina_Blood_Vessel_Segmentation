# **_Retina Blood Vessel Segmentation_**

## Dataset

This dataset contains a comprehensive collection of retinal fundus images, meticulously annotated for blood vessel segmentation. Accurate segmentation of blood vessels is a critical task in ophthalmology as it aids in the early detection and management of various retinal pathologies, such as diabetic retinopathy & glaucoma. <br>
The dataset comprises a total of 100 high-resolution retinal fundus images captured using state-of-the-art imaging equipment. Each image comes with corresponding pixel-level ground truth annotations indicating the exact location of blood vessels. These annotations facilitate the development and evaluation of advanced segmentation algorithms. <br>

- Dataset link on ![Kaggle](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)

![__results___47_0](https://github.com/user-attachments/assets/cc20f0ec-7f49-4a05-a108-e46fa25cd3ea)

## U-Net Architecture

U-Net is widely used in semantic segmentation because it excels at capturing fine-grained details and spatial context, thanks to its encoder-decoder architecture with skip connections. This design enables precise boundary delineation and efficient training even with a limited amount of labeled data. Moreover, U-Net's ability to preserve spatial information throughout the network significantly improves segmentation accuracy.

![image](https://github.com/user-attachments/assets/13771f61-6b66-4423-817e-7bdc143bf64e)

### Main Components:

1. Encoder (contracting path)
2. Bottleneck
3. Decoder (expansive path)
4. Skip Connections

<hr>

#### Encoder:

- Extract features from input images.
- Repeated 3x3 conv (valid conv) + ReLU layers.
- 2x2 max pooling to downsample (reduce spatial dimensions).
- Double channels with after the max pooling.

#### Bottleneck:

- Pivotal role in bridging the encoder and decoder.
- Capture the most abstract and high-level features from the input image.
- Serves as a feature-rich layer that condenses the spatial dimensions while preserving the semantic information.
- Enable the decoder to reconstruct the output image with high fidelity.
- The large number of channels in the bottleneck:
  <b> Balance the loss of spatial information due to down-sampling by enriching
  the feature space. </b>

#### Decoder:

- Repeated 3x3 conv (valid conv) + ReLU layers.
- Upsample using transpose convolution.
- Halves channels after transpose convolution.
- Successive blocks in decoder:
  <b> Series of gradual upsampling operations & gradual refinement helps in
  generating a high-quality segmentation map with accurate boundaries. </b>

#### Skip Connections:

- Preservation of Spatial Information because during the downsampling process, spatial information can be lost.
- Combining Low-level and High-level Features.
- Gradient Flow Improvement.
- Better Localization.
- Cropping is used in U-Net skip connections primarily due to the following reasons:
  - <b>Size Mismatch:</b> ensures that the sizes are compatible for concatenation.
  - <b>Aligning the central regions:</b> which contain more reliable information.

<hr>

#### Output:

- The final layer of the U-Net decoder typically has several filters equal to the number of classes, producing an output feature map for each class.
- The final layer of the U-Net can be a 1x1 convolution to map the feature maps to the desired number of output classes for segmentation.
- If there are C classes, the output will be of shape (H _ W _ C).
- Interpolation methods like bilinear or nearest-neighbor interpolation can be used at the final layer to adjust the output dimensions to match the input. This ensures that each pixel in the input image has a corresponding label in the output segmentation map.
- The softmax function is applied to each pixel location across all the channels


<hr>

## Model Evaluation

### Loss Function:

- The choice of loss function is crucial for training a U-Net model for blood vessel segmentation.
- The **Binary Cross-Entropy (BCE)** loss is commonly used for binary segmentation tasks, such as blood vessel segmentation.
- BCE loss is well-suited for pixel-wise classification problems where each pixel is classified as either a blood vessel or background. <br>

![image](https://github.com/user-attachments/assets/87862473-cb7f-4891-8a71-62fa850b23cf)

### Evaluation Metric:

- **IoU (Intersection over Union)**:
  - Measures the overlap between the predicted segmentation and the ground truth.
  - IoU is calculated as the ratio of the intersection area to the union area of the predicted and ground truth segmentation masks.
  - A higher IoU indicates better segmentation accuracy.

![image](https://github.com/user-attachments/assets/a040df1c-10bc-4678-9e42-26c2bc2d61f3)

## Hyperparameters:

| Job id | epochs | batch_size | learning_rate | val_IoU | val_loss | test_IoU | test_loss |
| -----: | -----: | ---------: | ------------: | ------: | -------: | -------: | --------: |
|  17855 |    100 |          4 |         1e-04 |  0.6163 |   0.1475 |        - |         - |
|  17857 |    100 |         16 |         1e-04 |  0.4087 |   0.2136 |        - |         - |
|  17931 |    200 |          4 |         1e-04 |  0.6783 |   0.1251 |   0.6779 |     0.125 |
|  17932 |    200 |          4 |         5e-05 |  0.6466 |   0.1361 |        - |         - |
|  17939 |    200 |          4 |         1e-04 |  0.6204 |   0.1457 |        - |         - |
|  17941 |    300 |          4 |         1e-04 |  0.6126 |   0.1551 |        - |         - |
|  17942 |    300 |          4 |         5e-05 |  0.5701 |   0.1618 |        - |         - |
|  18049 |    **400** |          **4** |         **1e-04** |  **0.7242** |   **0.0961** |   **0.6827** |    **0.1307** |


Experiments link on ![Comet](https://www.comet.com/youssefaboelwafa/retina-blood-vessel-segmentation/view/new/panels)

Number of GPUs used in the training is 4 GPUs

The best hyperparameters for my training after multiple experiments are:

- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Batch Size**: 4
- **Epochs**: 400

<br>

At epoch **373** the model has the best performance with: <br>

- **IoU score = 0.7242** <br>
- **validation loss = 0.0961** <br>

The model is saved to disk for future use.

## Inference:
![Untitled4](https://github.com/user-attachments/assets/179ea631-3a8b-46ea-b467-b8bae28b776d)
![Untitled3](https://github.com/user-attachments/assets/fa951288-3c9d-4b24-b922-d3cb4c1f5d4d)
![Untitled2](https://github.com/user-attachments/assets/f4742673-ed59-468e-a429-e4bdc127ad69)
![Untitled1](https://github.com/user-attachments/assets/330c41a1-14b7-4a83-ab05-0f191da1eb33)
![Untitled](https://github.com/user-attachments/assets/eda69ed1-4a41-474c-894e-9aa0c97e0027)

