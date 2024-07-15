# ***Retina Blood Vessel Segmentation***
## Dataset
This dataset contains a comprehensive collection of retinal fundus images, meticulously annotated for blood vessel segmentation. Accurate segmentation of blood vessels is a critical task in ophthalmology as it aids in the early detection and management of various retinal pathologies, such as diabetic retinopathy & glaucoma. <br>
The dataset comprises a total of 100 high-resolution retinal fundus images captured using state-of-the-art imaging equipment. Each image comes with corresponding pixel-level ground truth annotations indicating the exact location of blood vessels. These annotations facilitate the development and evaluation of advanced segmentation algorithms.

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
- If there are C classes, the output will be of shape (H * W * C).
- Interpolation methods like bilinear or nearest-neighbor interpolation can be used at the final layer to adjust the output dimensions to match the input. This ensures that each pixel in the input image has a corresponding label in the output segmentation map.
- The softmax function is applied to each pixel location across all the channels

<hr>
