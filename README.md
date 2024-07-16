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

```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec1 = self.conv_block(1024, 512)
        self.dec2 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec4 = self.conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.enc5(self.pool(x4))

        x = self.upconv4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)

        x = self.out(x)
        return x

```

<hr>

## PyTorch Model

