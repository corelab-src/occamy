import onnx
import onnxruntime

import os
import sys
import subprocess
import time
import torch
import torch.onnx
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

CDBUILD = os.environ.get('CORE_DNN_BUILD_PATH')
CORE_DNN = os.path.join(CDBUILD, "bin/core-dnn")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(CDBUILD, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

def execute_commands(cmds):
    subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True) #pretrained=True)

model.eval()
print(model)

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

torch.onnx.export(model, input_batch, "mobilenet.onnx", verbose=False)
onnx_model = onnx.load("mobilenet.onnx")
onnx.checker.check_model(onnx_model)
print("check model well formed")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_dnn",
    "--dnn-dealloc-opt=true",
    "--dnn-malloc-pool-opt=true",
    "mobilenet.onnx"])

input_batch_np = input_batch.cpu().numpy()

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("mobilenet.onnx")
ort_in = {ort_session.get_inputs()[0].name: input_batch_np}

sess = OMExecutionSession("./mobilenet.so")

for i in range(10):
    # using onnxruntime
    start = time.time()
    ort_out = ort_session.run(None, ort_in)
    diff = (time.time() - start)*1000
    if(i):
        ort_results.append(diff)
    else:
        print("ONNX RT first value : ", diff)

    # using pytorch
    model.to('cpu')
    torch.set_num_threads(4)
    start = time.time()
    torch_out = model(input_batch)
    diff = (time.time() - start)*1000
    if(i):
        torch_results.append(diff)
    else:
        print("Pytorch CPU first value : ", diff)

    # using occamy output
    start = time.time()
    occamy_out = sess.run(input_batch_np)
    diff = (time.time() - start)*1000
    if(i):
        occamy_results.append(diff)
    else:
        print("Occamy first value : ", diff)

    # using pytorch gpu
    if torch.cuda.is_available():
        start = time.time()
        torch.cuda.synchronize()
        torch.cuda.init()
        model.cuda()
        cuda_a = input_batch.cuda()
        cuda_out = model(cuda_a)
        diff = (time.time() - start)*1000

        if(i):
            cuda_results.append(diff)
        else:
            print("Pytorch GPU first value : ", diff)

    torch.cuda.empty_cache()

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

t = torch.ones(ort_out_t.size())
r = t.new_full(ort_out_t.size(), 100)
ort_out_round = (ort_out_t * r).round()
occamy_out_round = (occamy_out_t * r).round()
torch_out_round = (torch_out * r).round()
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
        cnt = 0
        for i in (torch_out_round.eq(cuda_out_round).reshape(-1) == False).nonzero():
          print(torch_out_round.reshape(-1)[i])
          print(cuda_out_round.reshape(-1)[i])
          cnt = cnt + 1
          if cnt >= 5:
            print ("print five wrong results only")
            break
    else:
      print("Correctness verified.")
  else:
    print("Not correct output! (occamy, torch different)")
    cnt = 0
    for i in (occamy_out_round.eq(torch_out_round).reshape(-1) == False).nonzero():
      cnt = cnt + 1
      if cnt >= 5:
        print ("print five wrong results only")
        break
else:
  print("Not correct output! (ort, occamy different)")
  cnt = 0
  for i in (ort_out_round.eq(occamy_out_round).reshape(-1) == False).nonzero():
    cnt = cnt + 1
    if cnt >= 5:
      print ("print five wrong results only")
      break

