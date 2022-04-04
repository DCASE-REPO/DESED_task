# Adapted from https://github.com/YuanGongND/ssast/blob/main/src/models/ast_models.py
# Samuele Cornell
# @Email cornellsamuele@gmail.com

# Original author:

# -*- coding: utf-8 -*-
# @Time    : 7/16/21 3:12 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

# the unified ast models for all pretraining/fine-tuning tasks.

import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
import random
from pathlib import Path

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                ):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if fstride != fshape or tstride != tshape:
            raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if model_size == 'tiny':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
            self.heads, self.depth = 3, 12
            self.cls_token_num = 2
        elif model_size == 'small':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
            self.heads, self.depth = 6, 12
            self.cls_token_num = 2
        elif model_size == 'base':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
            self.heads, self.depth = 12, 12
            self.cls_token_num = 2
        elif model_size == 'base_nokd':
            self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
            self.heads, self.depth = 12, 12
            self.cls_token_num = 1
        else:
            raise Exception('Model size must be one of tiny, small, base, base_nokd')

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        # SSL Pretraining Code
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim
        # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
        self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim),
                                                            requires_grad=False), nn.Parameter(
            torch.tensor(input_tdim), requires_grad=False)

        # masked patch classification (discriminative objective) layer
        # we use two layers for pretext task, but using a single layer has similar performance.
        # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
        # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
        self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                                        nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
        # masked patch reconstruction (generative objective) layer
        self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                                        nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
        self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

        # get the intermediate shape
        self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        num_patches = self.p_f_dim * self.p_t_dim
        self.num_patches = num_patches
        self.v.patch_embed.num_patches = num_patches
        print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
        print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
        print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
        print('pretraining number of patches={:d}'.format(num_patches))

        # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape),
                                   stride=(fstride, tstride))
        self.v.patch_embed.proj = new_proj

        # use trainable positional embedding
        new_pos_embed = nn.Parameter(
            torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

        # use a pretrained models for finetuning
        """
        
        """
    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)



    # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # equation (1) of the ssast paper
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])
            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def forward(self, x, task="ft", cluster=False, mask_patch=0):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        if x.shape[-1] > self.input_tdim:
            raise RuntimeError("input longer than max len supported by this model, "
                               "you may consider to do inference over chunks with {} frames "
                               "of the original signal".format(self.input_tdim))

        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_cls':
            return self.finetuningcls(x)
        elif task == "ft":
            return self.finetuning(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

    def finetuningcls(self, x):
        # not used currently see original paper
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed[:, :x.shape[1]]
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        return x

    def finetuning(self, x):

        B = x.shape[0]
        x = self.v.patch_embed(x)

        x = x + self.v.pos_embed[:, :x.shape[1]]
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        return x


    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path=None, *args, **kwargs):
        """

        """

        if pretrained_model_conf_or_path == None:
            raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
        sd = torch.load(pretrained_model_conf_or_path, map_location="cpu")
        # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
        try:
            p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], \
                                 sd['module.v.patch_embed.proj.weight'].shape[3]
            p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
        except:
            raise ValueError(
                'The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')

        print('now load a SSL pretrained models from ' + pretrained_model_conf_or_path)
        model_size = Path(pretrained_model_conf_or_path).stem.split("-")[1].lower()
        audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                               input_fdim=p_input_fdim, input_tdim=p_input_tdim,
                               model_size=model_size)
        audio_model.load_state_dict(sd, strict=False)

        audio_model.v = audio_model.v
        audio_model.original_embedding_dim = audio_model.v.pos_embed.shape[2]
        audio_model.cls_token_num = audio_model.cls_token_num


        f_dim, t_dim = audio_model.get_shape(audio_model.fstride, audio_model.tstride, audio_model.input_fdim,
                                             audio_model.input_tdim, audio_model.fshape, audio_model.tshape)
        # patch array dimension during pretraining
        p_f_dim, p_t_dim = audio_model.p_f_dim, audio_model.p_t_dim
        num_patches = f_dim * t_dim
        p_num_patches = p_f_dim * p_t_dim
        audio_model.v.patch_embed.num_patches = num_patches
        print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(audio_model.fstride, audio_model.tstride))
        print('fine-tuning number of patches={:d}'.format(num_patches))

        new_pos_embed = audio_model.v.pos_embed[:, audio_model.cls_token_num:, :].detach().reshape(1, p_num_patches,
                                                                                     audio_model.original_embedding_dim).transpose(
            1, 2).reshape(1, audio_model.original_embedding_dim, p_f_dim, p_t_dim)
        # cut or interpolate the positional embedding
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[:, :, :,
                            int(p_t_dim / 2) - int(t_dim / 2): int(p_t_dim / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[:, :,
                            int(p_f_dim / 2) - int(f_dim / 2): int(p_f_dim / 2) - int(f_dim / 2) + t_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

        new_pos_embed = new_pos_embed.reshape(1, audio_model.original_embedding_dim, num_patches).transpose(1, 2)
        audio_model.v.pos_embed = nn.Parameter(
            torch.cat([audio_model.v.pos_embed[:, :audio_model.cls_token_num, :].detach(), new_pos_embed], dim=1))

        return audio_model


if __name__ == '__main__':

    # fine-tuning stage
    # now you have a labeled dataset you want to finetune AST on
    # suppose the avg length is 100 frames (1s) and there are 35 classes
    # the fshape and tshape must be same in pretraining and finetuning
    # but fstride and tstride can be different in pretraining and finetuning
    # using smaller strides improves the performance but also increase the computational overhead
    # set pretrain_stage as False since now is in the finetuning stage
    # provide the path of the pretrained model you want to load
    input_tdim = 1000  # fine-tuning data length can be different with pretraining data length
    # # alternatively, use a frame based AST model
    ast_mdl = ASTModel.from_pretrained('/tmp/SSAST-Base-Patch-400.pth')

    # do finetuning, see src/traintest.py for our finetuning code
    test_input = torch.zeros([1, input_tdim, 128])
    prediction = ast_mdl(test_input)
    # output should in shape [batch_size, label_dim]
    print(prediction.shape)
