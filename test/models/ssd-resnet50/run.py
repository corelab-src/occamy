import os
import torch

#forcing thread count
os.environ["OMP_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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

CDBUILD = os.environ.get('CORE_DNN_BUILD_PATH')
CORE_DNN = os.path.join(CDBUILD, "bin/core-dnn")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(CDBUILD, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

def execute_commands(cmds):
  subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

precision = 'fp32'
#ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, map_location=torch.device('cpu'))
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

uris = [
  'http://images.cocodataset.org/val2017/000000397133.jpg',
  'http://images.cocodataset.org/val2017/000000037777.jpg',
  'http://images.cocodataset.org/val2017/000000252219.jpg'
]

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == 'fp16')

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    ssd_model.to('cuda')

with torch.no_grad():
    detections_batch = ssd_model(tensor)

ssd_model.eval()

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()

torch.onnx.export(ssd_model, tensor, "ssd-resnet50.onnx", verbose=True)
onnx_model = onnx.load("ssd-resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("check model well formed")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_dnn",
    "--dnn-dealloc-opt=true",
    "--dnn-malloc-pool-opt=true",
    "ssd-resnet50.onnx"])

input_batch_np = tensor.cpu().numpy()

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("ssd-resnet50.onnx")
ort_in = {ort_session.get_inputs()[0].name: input_batch_np}

sess = OMExecutionSession("./ssd-resnet50.so")

for i in range(100):
    # using onnxruntime
    start = time.time()
    ort_out = ort_session.run(None, ort_in)
    diff = (time.time() - start)*1000
    if(i):
        ort_results.append(diff)
    else:
        print("first value : ", diff)

    # using pytorch
    ssd_model.to('cpu')
    cpu_a = tensor.to('cpu')
    torch.set_num_threads(4)
    start = time.time()
    with torch.no_grad():
        torch_out = ssd_model(cpu_a)
    diff = (time.time() - start)*1000
    if(i):
        torch_results.append(diff)
    else:
        print("first value : ", diff)

    # using occamy output
    start = time.time()
    occamy_out = sess.run(input_batch_np)
    diff = (time.time() - start)*1000
    if(i):
        occamy_results.append(diff)
    else:
        print("first value : ", diff)

    # using pytorch gpu
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.init()
        ssd_model.cuda()
        cuda_a = tensor.cuda()
        start = time.time()
        cuda_out = ssd_model(cuda_a)
        diff = (time.time() - start)*1000
        if(i):
            cuda_results.append(diff)
        else:
            print("first value : ", diff)

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


isCorrect = []

for i, e in enumerate(ort_out):
  ort_out_t = torch.tensor(e)
  occamy_out_t = torch.tensor(occamy_out[i])
  if i < len(ort_out)-1:
    torch_out_t = torch_out[0][i].cpu()
    cuda_out_t = cuda_out[0][i].cpu()
  else:
    torch_out_t = torch_out[1][0].cpu()
    cuda_out_t = cuda_out[1][0].cpu()
  t = torch.ones(ort_out_t.size())
  r = t.new_full(ort_out_t.size(), 10) # sensitivity
  ort_out_round = (ort_out_t * r).round()
  occamy_out_round = (occamy_out_t * r).round()
  torch_out_round = (torch_out_t * r).round()
  if torch.cuda.is_available():
    cuda_out_round = (cuda_out_t * r).round()

  # correctness
  if torch.all(ort_out_round.eq(occamy_out_round).reshape(-1)):
    if torch.all(occamy_out_round.eq(torch_out_round).reshape(-1)):
      if torch.cuda.is_available():
        if torch.all(torch_out_round.eq(cuda_out_round).reshape(-1)):
          print("Correctness verified.")
          isCorrect.append(True)
        else:
          print("Not correct output! (torch, cuda different)")
          isCorrect.append(False)
  #        print(torch_out_round)
  #        print(cuda_out_round)
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
        isCorrect.append(True)
    else:
      print("Not correct output! (occamy, torch different)")
      isCorrect.append(False)
  #    print(occamy_out_round)
  #    print(torch_out_round)
      cnt = 0
      for i in (occamy_out_round.eq(torch_out_round).reshape(-1) == False).nonzero():
        print(occamy_out_round.reshape(-1)[i])
        print(torch_out_round.reshape(-1)[i])
        cnt = cnt + 1
        if cnt >= 5:
          print ("print five wrong results only")
          break
  else:
    print("Not correct output! (ort, occamy different)")
    isCorrect.append(False)
    cnt = 0
    for i in (ort_out_round.eq(occamy_out_round).reshape(-1) == False).nonzero():
      print(ort_out_round.reshape(-1)[i])
      print(occamy_out_round.reshape(-1)[i])
      cnt = cnt + 1
      if cnt >= 5:
        print ("print five wrong results only")
        break

if all(isCorrect):
  print("All rows are correctly verified.")
else:
  print("Correctness verification failed.")

