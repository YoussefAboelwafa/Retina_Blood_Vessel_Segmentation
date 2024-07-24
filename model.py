import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class LitUnet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate):
        super(LitUnet, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

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

        self.train_loss_sum = 0.0
        self.train_iou_sum = 0.0
        self.train_batches = 0
        
        self.val_loss_sum = 0.0
        self.val_iou_sum = 0.0
        self.val_batches = 0

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def common_step(self, batch, batch_idx):
        image, mask = batch
        pred = self.forward(image)
        loss = self.criterion(pred, mask)
        pred = torch.sigmoid(pred)
        mask = mask.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask, mode="binary", threshold=0.5)  # type: ignore
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        return loss, iou_score

    def training_step(self, batch, batch_idx):
        loss, iou_score = self.common_step(batch, batch_idx)
        self.train_loss_sum += loss
        self.train_iou_sum += iou_score
        self.train_batches += 1
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = self.train_loss_sum / self.train_batches
        avg_train_iou = self.train_iou_sum / self.train_batches
        self.log('train_iou', avg_train_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', avg_train_loss, on_epoch=True, prog_bar=True, logger=True)
        self.train_loss_sum = 0.0
        self.train_iou_sum = 0.0
        self.train_batches = 0
        
        
    def validation_step(self, batch, batch_idx):
        loss, iou_score = self.common_step(batch, batch_idx)
        self.val_loss_sum += loss
        self.val_iou_sum += iou_score
        self.val_batches += 1
        return loss
    
    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_sum / self.val_batches
        avg_val_iou = self.val_iou_sum / self.val_batches
        self.log('val_iou', avg_val_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.val_loss_sum = 0.0
        self.val_iou_sum = 0.0
        self.val_batches = 0  

    def test_step(self, batch, batch_idx):
        loss, iou_score = self.common_step(batch, batch_idx)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_iou",
            iou_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
