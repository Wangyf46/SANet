import os
import glob
import time
import numpy as np
import torch
import shutil
import ipdb
import argparse
from PIL import Image

from src.model import *
from src.utils import *
from src.dataset import *

parser = argparse.ArgumentParser(description = "PyTorch SANet Test")
parser.add_argument("--gpu", default = "8", type = str, 
                    help = "GPU id to use.")
parser.add_argument("--dataset_path", 
    default=("/data/wangyf/datasets/Shanghai/part_B_final/test_data/images/"),
                   type = str)
parser.add_argument("--output_dir",
                    default = "/data/wangyf/Output/SANet/", 
                    type = str) 

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

start = time.time()

output_dir = args.output_dir + "s_300_0320model_best/"  
part_B_test =args.dataset_path
path_sets = [part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        img_paths.append(img_path)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  #delete and mkdir
    os.mkdir(output_dir)
else:
    os.mkdir(output_dir)

file_results = output_dir + "B_test_result.txt"  # todo
model = SANet()
#model = nn.DataParallel(model)  # multi-gpu, actually only ong gpu execute
checkpoint = torch.load("s_300_0320model_best.pth.tar")  ## TODO
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda().eval()
mae = 0.0
mse = 0.0
f = open(file_results, 'w')
for i in range(len(img_paths)):
    print(img_paths[i])
    img = Image.open(img_paths[i]).convert('RGB')                # size = (W,H)
    label_count, gt_density_map = gt_label(img_paths[i])
    gt_count = np.sum(gt_density_map)
    et_density_map = et_data(img, model)
    et_count = et_density_map.sum()

    ## accuracy and robustness
    diff = gt_count - et_count
    mae += abs(diff)
    mse += diff ** 2

    ## data normalization
    gt_density_map = gt_density_map * 255 / np.max(gt_density_map)
    et_density_map = et_density_map * 255 / np.max(et_density_map)
   
    ## quality assessment
    PSNR, _ = get_quality_psnr(gt_density_map, et_density_map)
    SSIM = get_quality_ssim(gt_density_map, et_density_map)

    ## visualization and save
    path = img_paths[i].replace(
        "/data/wangyf/datasets/Shanghai/part_B_final/test_data/images/IMG_", "")
    save_path = output_dir + "img_gt_et_" + path
    visualization_save(save_path, 
                       img, gt_density_map, et_density_map,
                       label_count, gt_count, et_count, PSNR, SSIM)
    f.write("Image_{path};  "
            "label_count:{label_count:.2f}"
            "gt_count:{gt_count:.2f};  "
            "et_count:{et_count:.2f};  "
            "diff:{diff:.2f}  "
            "PSNR:{PSNR:.2f}; "
            "SSIM:{SSIM:.2f};\n "
            .format(path = path, 
                    label_count = label_count, 
                    gt_count = gt_count, 
                    et_count = et_count,
                    diff = diff, 
                    PSNR = PSNR, 
                    SSIM = SSIM))
end = time.time()
second = end - start
print("elapsed time:{elapsed_time:.4f};  "
      "fps:{fps:.4f};   "
      "mae:{mae:.4f};   "
      "mse:{mse:.4f};\n "
      .format(elapsed_time = second, 
              fps = len(img_paths) / second,
              mae = mae / len(img_paths), 
              mse = np.sqrt(mse / len(img_paths))))
f.write("elapsed time:{elapsed_time:.4f};  "
        "fps:{fps:.4f};   "
        "mae:{mae:.4f};   "
        "mse:{mse:.4f};\n "
        .format(elapsed_time = second, 
                fps = len(img_paths) / second,
                mae = mae / len(img_paths), 
                mse = np.sqrt(mse / len(img_paths))))
f.close()

