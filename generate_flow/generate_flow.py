import argparse
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from core.network import RAFTGMA
from core.utils.utils import InputPadder
from core.utils.frame_utils import *
import time
from tqdm.contrib import tzip


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def normalize(x):
    return x / (x.max() - x.min())

def check_img(path):
    return path[-4:] == '.png'

def make_dir(p):
    if not osp.exists(p):
        os.makedirs(p)
    return


def generate_flows(args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print("Loaded checkpoint at {:s}".format(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    for dstype in ['train', 'test']:
        scene_list = sorted(os.listdir(osp.join(args.data_path, dstype)))
        for scene in scene_list:
            image_list = sorted(os.listdir(osp.join(args.data_path, dstype, scene)))

            make_dir(osp.join(args.save_path, dstype, scene))

            print("Estimated flow for scene {:s} using model {:s}".format(scene, args.model))
            with torch.no_grad():
                for imfile1, imfile2 in tzip(image_list[:-args.interval], image_list[args.interval:]):
                    flow_save_path = osp.join(args.save_path, dstype, scene, imfile1[:-4]+'.flo')
                    if osp.exists(flow_save_path):
                        continue
                    if not (check_img(imfile1) and check_img(imfile2)):
                        continue
                    
                    st = time.time()
                    im_path1 = osp.join(args.data_path, dstype, scene, imfile1)
                    im_path2 = osp.join(args.data_path, dstype, scene, imfile2)
                    image1 = load_image(im_path1)
                    image2 = load_image(im_path2)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
                    flow_out = padder.unpad(flow_up)
                    flow_out = flow_out.squeeze(dim=0).permute([1,2,0]).detach().cpu().numpy()

                    writeFlow(flow_save_path, flow_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rssf-path', default='/home/data/rzhao/rssf')
    parser.add_argument('--interval', '-int', type=int, default=20)
    parser.add_argument('--model', default='./checkpoints/gma-sintel.pth', help="restore checkpoint")
    parser.add_argument('--model_name', default="GMA", help="define model name")
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--cuda', '-cu', type=str, default='0')
    args = parser.parse_args()

    args.interval = args.interval // 20
    args.data_path = osp.join(args.rssf_path, 'imgs')
    args.save_path = osp.join(args.rssf_path, 'flow{:d}'.format(args.interval))
    DEVICE = 'cuda'

    generate_flows(args)
