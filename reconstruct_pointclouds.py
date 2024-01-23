import numpy as np
import os
import configargparse
import open3d as o3d
import cv2
import torch
import imageio
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hwf = [720, 1280, 686.926]
print("hwf = [720, 1280, 686.926] ")

def load_rgb_images(path):
    return imageio.v2.imread(path)

def load_depth_images(path):

    temp = imageio.v2.imread(path)
    temp = np.clip(temp, 0, 255)
    return temp.astype(np.uint8)

def load_depth_npy(path):
    return np.load(path)

def list_given_ext(dir, ext='.png'):

    return [f for f in os.listdir(dir) if f.endswith(ext)]

def reconstruct_pointclouds(rgb_np, depth_np, depth_filter=None, verbose=True, crop_left_size=0):

    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]
        depth_np = depth_np[:, crop_left_size:]

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

    if verbose:
        print('min depth:', depth_np.min(), 'max depth:', depth_np.max())

    rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth_np)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, depth_im, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(hwf[1], hwf[0], hwf[2], hwf[2], hwf[1] / 2, hwf[0] / 2)
    )

    return pcd

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/icassp/boxs', help='root_path')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument("--depth_smoother", action='store_true', help='apply bilateral filtering on depth maps?')
    parser.add_argument("--depth_smoother_d", type=int, default=32, help='diameter of bilateral filter for depth maps')
    parser.add_argument("--depth_smoother_sv", type=float, default=64)
    parser.add_argument("--depth_smoother_sr", type=float, default=32)
    parser.add_argument('--crop_left_size', type=int, default=0, help='crop left size')
    parser.add_argument("--no_pc_saved", action='store_true', help='donot save reconstructed point clouds?')
    parser.add_argument('--out_postfix', type=str, default='', help='the postfix append to the output directory name')
    parser.add_argument('--data_type', type=str, default=None, help='the postfix append to the output directory name')

    args = parser.parse_args()
    # depth filter
    if args.depth_smoother:
        depth_smoother = (args.depth_smoother_d,
                          args.depth_smoother_sv, args.depth_smoother_sr)
    else:
        depth_smoother = None
    # reconstruct pointclouds
    print('reconstructing point clouds...')

    if not args.no_pc_saved:
        out_dir = os.path.join(os.path.dirname(args.root_path), 'pointclouds')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        parser.write_config_file(args, [os.path.join(out_dir, 'args.txt')])

    img_names = [i for i in sorted(list_given_ext(args.root_path), key=lambda x: int(x.split('.')[0])) if i.endswith('.png')]
    depth_names = [i for i in sorted(list_given_ext(args.root_path, ext='.npy'), key=lambda x: int(x.split('.')[0][5:])) if i.endswith('.npy')]
    print(img_names, depth_names)
    for i in tqdm(range(len(img_names))):
        img = load_rgb_images(os.path.join(args.root_path, img_names[i]))
        depth = load_depth_npy(os.path.join(args.root_path, depth_names[i])).astype(np.float32)
        pcd = reconstruct_pointclouds(img, depth, args.vis_rgbd, depth_smoother, args.verbose, args.crop_left_size)

