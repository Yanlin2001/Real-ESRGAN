import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """
        Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        接受来自数据加载器的数据，然后添加二阶降级以获得 LQ 图像。
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)

            # ++ lq data synthesis ++
            self.usp = data['lq'].to(self.device) # undersampled

            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.usp.size()[2:4]

            # ---------- The first degradation process (第一个退化过程) ------------------ #
            # blur
            out = filter2D(self.usp, self.kernel1)
            # ------------- The second degradation process (第二个退化过程) ------------- #

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
        '''
        lq_image = transforms.ToPILImage()(self.lq.cpu())  # Convert PyTorch tensor to PIL Image
        plt.imshow(lq_image)
        plt.show()
        '''
        print("self.lq.shape: ", self.lq.shape)
        import datetime
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms

        # Assuming self.lq is a PyTorch tensor
        sample_index = 0  # Choose the index of the sample you want to visualize
        lq_image = transforms.ToPILImage()(self.lq[sample_index].cpu())  # Convert to PIL Image

        # Get current time
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Save image with current time as filename
        save_path = f"/kaggle/working/lq_image_{current_time}.png"
        lq_image.save(save_path)

        print(f"Image saved at: {save_path}")



    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
