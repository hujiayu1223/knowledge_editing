import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import matplotlib.colors as mcolors
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.Halle_Editor.halle_editor import hall_editor
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import json
import torch.distributed as dist

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    # "minigpt4": " <Img><ImageHere></Img> <question>",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True





parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--beam", type=int)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
parser.add_argument("--prompt_t", type=int, default=0)
parser.add_argument("--results_save_dir", type=str, default="./edited_model", help="model save path")
args = parser.parse_known_args()[0]





# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')
# ================多GPU运行===============
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)
 # 多gpu运行
if args.prompt_t:
    model.eval()
else:
    model.train() #在进行prompt learning的时候需要冻结其他参数
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

# data_path=../halle/playground/data/coco/val2017
img_files = os.listdir(args.data_path) #要取1-3600，4100-500的片段
# random.shuffle(img_files)
# img_files = img_files[:3600]+img_files[4100:]
# random.seed() # 生成不同的随机种子
# random.shuffle(img_files)

# with open('../halle/playground/data/coco/annotations/instances_val2017.json', 'r') as f:
#     lines = f.readlines()
# coco_anns = json.loads(lines[0])

# with open('dataset/train/halle_2000.jsonl', 'r') as f:
#     lines = f.readlines()
target_dict={}
halle_dict={}
id_list=[]
qu_dict={}
img_dict = {}

val_img_dict = {}

with open("dataset/train/halle_2000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line) 
        halle_dict[data["image_id"]]=data["caption"]
        id_list.append(data["image_id"])
with open("dataset/train/image_2000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line) 
        target_dict[data["image_id"]]=data["caption"]




base_dir  = "./log/chair/" 
edit_dir = "./log/edit/" + args.model
output_dir = "./dataset/test/"

if not os.path.exists(base_dir):
    os.mkdir(base_dir)
# for img_id in tqdm(range(500)):
random_numbers = random.sample(range(0, len(img_files)), 500)

id = []
img = {}
con_id = []
con_img = {}
for img_id in tqdm(id_list[:1500]):
    # qu_image = "Please describe this image in detail based on the specific, visible information. Focus on concrete elements such as colors, shapes, objects, positions, textures, and expressionss."
    qu_image = "Please describe this image in detail based on the specific, visible information and avoid any imagination which is not in the image."
    qu_halle_1 = "Please describe this image in detail based on extensive internal knowledge and understanding of patterns, reasonable speculation like relationship about the scene or the existence and attribute of objects is allowed."
    qu_halle_2 = "Please describe this image in detail based on extensive internal knowledge and understanding of patterns."
    qu_halle_3 = "Please describe this image in detail based on extensive internal knowledge and imagination of existing objects."
    qu_2 = "Please describe this image in detail, reasonable speculation about the scene or the purpose of objects is allowed, but do not deviate from the context of the image. "
    qu_norm = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    qu_image = template.replace("<question>", qu_image)
    qu_halle = template.replace("<question>", qu_halle_1)
    qu_2 = template.replace("<question>", qu_2)
    qu_norm = template.replace("<question>", qu_norm)
    question = [qu_image, qu_halle]
    qu = {"qu_image": qu_image, "qu_halle": qu_halle,"qu_norm":qu_norm} 
    # for img_id in range(5): # 用1组先试试
    if img_id in id_list[:1500]:
        img_file = str(img_id).zfill(12)+".jpg"
    else:
        img_file = str(img_files[img_id])
        img_id = int(img_file.split(".jpg")[0][-6:])
    # img_file = "VizWiz_test_"+str(img_id).zfill(8)+".jpg"
    img_save = {}
    img_save["image_id"] = img_id
    image_path = args.data_path + img_file
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    # image = add_diffusion_noise(image, -500).to(device)
    img[img_id] = image
    id.append(int(img_id))
    # with torch.inference_mode():
    #     with torch.no_grad():
    #         image = norm(image).to(device)
    #         _,output = model.generate(
    #             {"image": image, "prompt": qu_norm},
    #             use_nucleus_sampling=args.sample,
    #             num_beams=args.beam,
    #             max_new_tokens=512,
    #             output_attentions=True,
    #             opera_decoding=False,
    #             scale_factor=args.scale_factor,
    #             threshold=args.threshold,
    #             num_attn_candidates=args.num_attn_candidates,
    #             penalty_weights=args.penalty_weights,
    #             generate=True, #查看注意力是否发生变化
    #         )
    # attn_square = attn_square[0].cpu().numpy()
    # start = index["image_start"]
    # end = index["image_end"]
    # hallu_start = index["response_start"]+index["hallu_idx"]
    # hallu_end = index["response_start"]+index["hallu_idx"]+index["hallu_len"]

    # print(index["hallu_len"])
    # print(np.sum(attn_square[hallu_end,34:610]))
    # exit()
    # cmap = mcolors.ListedColormap(['#FFFFFF', '#FF0000'])  # 定义颜色映射，这里使用红色从白色开始
    # # norm = mcolors.BoundaryNorm(np.linspace(0, 1, 256), cmap.N)
    # plt.imshow(attn_square[index["response_start"]:hallu_end,index["response_start"]:hallu_end], cmap='hot')  # cmap参数指定颜色映射
    # plt.title(str(img_id)+' Ouput Attn Square Visualization Layer31')
    # filename = str(img_id)+'_textOnly.png'
    # plt.savefig(filename,dpi=1500,format='png')
    # img_save["caption"] = output[0]
    # with open(os.path.join(base_dir, 'Vizwiz_ours_500.jsonl'), "a") as f:
    #     json.dump(img_save, f)
    #     f.write('\n')
requests = {"id": id, "image": img, "prompt": qu, "target": target_dict, "halle": halle_dict,"val":val_img_dict}
#unlearning
hparams = 'minigpt4/Halle_Editor/vicuna-7b.yaml'
# 读取prompt的向量参数如果没有的话先构建一个
# prompt_learner = prompt_tuning(hparams,model, requests,device)
# prompt_vec = prompt_learner.tuning(args)
halle_editor = hall_editor(hparams,model, requests,device)
edited_model = halle_editor.edit(args)
save_path = f'{args.results_save_dir}/llava-1.5-ours-chat'
# exit()
edited_model.save_pretrained(save_path)
print(f"edited model is saved in {save_path}")
exit()
# test

    # out,attn_square,dict_embed,responce_start = model.generate(
    #     {"image": norm(image), "prompt": question},
    #     use_nucleus_sampling=args.sample,
    #     num_beams=args.beam,
    #     max_new_tokens=512,
    #     output_attentions=True,
    #     opera_decoding=False,
    #     scale_factor=args.scale_factor,
    #     threshold=args.threshold,
    #     num_attn_candidates=args.num_attn_candidates,
    #     penalty_weights=args.penalty_weights,
    # )

    # print(dict_embed)
    # cmap = mcolors.ListedColormap(['#FFFFFF', '#FF0000'])  # 定义颜色映射，这里使用红色从白色开始
    # norm = mcolors.BoundaryNorm(np.linspace(0, 1, 256), cmap.N)
    # plt.imshow(attn_square, cmap='hot')  # cmap参数指定颜色映射
    # plt.title(str(img_id)+' Ouput Attn Square Visualization Layer31')
    # filename = str(img_id)+'_Layer31.png'
    # plt.savefig(filename,dpi=1500,format='png')
    #
    # plt.imshow(attn_square[-150:,dict_embed['image_start']:], cmap='hot')  # cmap参数指定颜色映射
    # plt.title('Ouput Attn Square Visualization Layer31')
    # filename = str(img_id)+'_Layer31_question_generation.png'
    # plt.savefig(filename,dpi=1500,format='png')
    #
    # plt.imshow(attn_square[:,dict_embed['image_start']:dict_embed['image_end']], cmap='hot')  # cmap参数指定颜色映射
    # plt.title('Ouput Attn Square Visualization Layer31')
    # filename = str(img_id)+'_Layer31_image.png'
    # plt.savefig(filename,dpi=1500,format='png')
    #
    # plt.imshow(attn_square[:,responce_start:], cmap='hot')  # cmap参数指定颜色映射
    # plt.title('Ouput Attn Square Visualization Layer31')
    # filename = str(img_id)+'_Layer31_generation.png'
    # plt.savefig(filename,dpi=1500,format='png')
    # dump metric file

    # with open(os.path.join(base_dir, 'ours-prior_focus-num_beam_{}-num_can_{}.jsonl'.format(args.beam,args.num_attn_candidates)), "a") as f:
    #     json.dump(img_save, f)
    #     f.write('\n')

# python chair_eval.py --model llava-1.5 --data_path ../halle/playground/data/coco/val2017/ --gpu-id 1 --beam 2 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1 --prompt qu_norm



