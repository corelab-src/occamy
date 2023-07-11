import os
import torch
#forcing thread count
os.environ["OMP_NUM_THREADS"] = "1"

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

import onnx
import onnxruntime

import sys
import subprocess
import time
import torch.onnx
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
# from SRGAN-PyTorch (https://github.com/dongheehand/SRGAN-PyTorch)
from dataset import *
from srgan_model import Generator, Discriminator

CDBUILD = os.environ.get('CORE_DNN_BUILD_PATH')
CORE_DNN = os.path.join(CDBUILD, "bin/core-dnn")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(CDBUILD, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

def execute_commands(cmds):
  subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

_LR_path = "./DIV2K_LR_IMAGE"
_Generator_path = "./model/SRGAN.pt"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
dataset = testOnly_data(LR_path = _LR_path, in_memory = False, transform = None)
loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)

model = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16)
model.load_state_dict(torch.load(_Generator_path))
model = model.to(device)
model.eval()

for i, te_data in enumerate(loader):
  lr = te_data['LR'].to(device)
  break # infer only one image

torch.onnx.export(model, lr, "srgan.onnx", verbose=True)
onnx_model = onnx.load("srgan.onnx")
onnx.checker.check_model(onnx_model)
print("check model well formed")

print("Compiling model with Occamy ...", end='')
execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_dnn",
    "--dnn-dealloc-opt=true",
    "--dnn-malloc-pool-opt=true",
    "srgan.onnx"])
print(" : compile done!")

input_batch_np = lr.cpu().numpy()

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("srgan.onnx")
ort_in = {ort_session.get_inputs()[0].name: input_batch_np}

sess = OMExecutionSession("./srgan.so")

for i in range(2):
    # using onnxruntime
    start = time.time()
    ort_out, _ = ort_session.run(None, ort_in)
    ort_out =  ort_out[0]
    ort_out = (ort_out + 1.0) / 2.0
    ort_out = ort_out.transpose(1,2,0)
    ort_res = Image.fromarray((ort_out * 255.0).astype(np.uint8))
    ort_res.save('./result/ort_res_0000.png')
    diff = (time.time() - start)*1000
    if(i):
        ort_results.append(diff)
    else:
        print("first value : ", diff)

    # using pytorch
    lr = lr.to('cpu')
    model.to('cpu')
    start = time.time()
    with torch.no_grad():
        torch_out, _ = model(lr)
        torch_out = torch_out[0].cpu().numpy()
        torch_out = (torch_out + 1.0) / 2.0
        torch_out = torch_out.transpose(1,2,0)
        result = Image.fromarray((torch_out * 255.0).astype(np.uint8))
        result.save('./result/res_%04d.png'%i)
    diff = (time.time() - start)*1000
    if(i):
        torch_results.append(diff)
    else:
        print("first value : ", diff)

    # using pytorch gpu
    if torch.cuda.is_available():
        start = time.time()
        torch.cuda.synchronize()
        torch.cuda.init()
        model.cuda()
        cuda_a = lr.cuda()
        cuda_out, _ = model(cuda_a)
        cuda_out = cuda_out[0].cpu().detach().numpy()
        cuda_out = (cuda_out + 1.0) / 2.0
        cuda_out = cuda_out.transpose(1,2,0)
        cuda_res = Image.fromarray((cuda_out * 255.0).astype(np.uint8))
        cuda_res.save('./result/cuda_res_%04d.png'%i)
        diff = (time.time() - start)*1000
        del cuda_a
        del cuda_out
        del _
    if(i):
        cuda_results.append(diff)
    else:
        print("first value : ", diff)

    torch.cuda.empty_cache()
    # using occamy output
    start = time.time()
    occamy_out, _ = sess.run(input_batch_np)
    occamy_out =  occamy_out[0]
    occamy_out = (occamy_out + 1.0) / 2.0
    occamy_out = occamy_out.transpose(1,2,0)
    occamy_res = Image.fromarray((occamy_out * 255.0).astype(np.uint8))
    occamy_res.save('./result/occamy_res_0000.png')
    diff = (time.time() - start)*1000
    if(i):
        occamy_results.append(diff)
    else:
        print("first value : ", diff)

print("\n\n================ ONNX Runtime =================")
print("Percentile 90 : ", np.percentile(ort_results, 90))
print("Percentile 95 : ", np.percentile(ort_results, 95))
print("Percentile 99 : ", np.percentile(ort_results, 99))
print("Average : ", np.average(ort_results))

print("================ Pytorch =================")
print("Percentile 90 : ", np.percentile(torch_results, 90))
print("Percentile 95 : ", np.percentile(torch_results, 95))
print("Percentile 99 : ", np.percentile(torch_results, 99))
print("Average : ", np.average(torch_results))

print("================ Pytorch GPU =================")
print("Percentile 90 : ", np.percentile(cuda_results, 90))
print("Percentile 95 : ", np.percentile(cuda_results, 95))
print("Percentile 99 : ", np.percentile(cuda_results, 99))
print("Average : ", np.average(cuda_results))

print("================ Occamy =================")
print("Percentile 90 : ", np.percentile(occamy_results, 90))
print("Percentile 95 : ", np.percentile(occamy_results, 95))
print("Percentile 99 : ", np.percentile(occamy_results, 99))
print("Average ", np.average(occamy_results), "\n\n")

ort_out_t = torch.tensor(ort_out)[0]
occamy_out_t = torch.tensor(occamy_out)[0]
torch_out_t = torch.tensor(torch_out)[0]
print(ort_out_t.eq(occamy_out_t))
print(ort_out_t.size())

t = torch.ones(ort_out_t.size())
r = t.new_full(ort_out_t.size(), 100)
ort_out_round = (ort_out_t * r).round()
occamy_out_round = (occamy_out_t * r).round()
torch_out_round = (torch_out_t * r).round()
if torch.cuda.is_available():
 cuda_out_round = (cuda_out.to("cpu") * r).round()

# correctness
if torch.all(ort_out_round.eq(occamy_out_round).reshape(-1)):
    if torch.all(occamy_out_round.eq(torch_out_round).reshape(-1)):
        if torch.cuda.is_available():
            if torch.all(torch_out_round.eq(cuda_out_round).reshape(-1)):
                print("Correctness verified.")
            else:
                print("Not correct output! (torch, cuda different)")
                print(torch_out_round)
                print(cuda_out_round)
        for i in (torch_out_round.eq(cuda_out_round).reshape(-1) == False).nonzero():
            print(torch_out_round.reshape(-1)[i])
            print(cuda_out_round.reshape(-1)[i])
        else:
            print("Correctness verified.")
            print("Correctness verified.")
    else:
        print("Not correct output! (occamy, torch different)")
        for i in (occamy_out_round.eq(torch_out_round).reshape(-1) == False).nonzero():
            print(occamy_out_round.reshape(-1)[i])
            print(torch_out_round.reshape(-1)[i])
else:
    print("Not correct output! (ort, occamy different)")
    for i in (ort_out_round.eq(occamy_out_round).reshape(-1) == False).nonzero():
        print(ort_out_round.reshape(-1)[i])
        print(occamy_out_round.reshape(-1)[i])

