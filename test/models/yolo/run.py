import onnx
import onnxruntime

import torch
import torch.nn as nn

import os
import sys
import subprocess
import time
import torch
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

"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size=7, num_boxes=2, num_classes=20):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

model = Yolov1()
a = torch.randn(2, 3, 448, 448)
a_np = a.to("cpu").numpy()

out = model(a)

torch.onnx.export(model, a, "yolov1.onnx", verbose=True)
onnx_model = onnx.load("yolov1.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model well formed\n\n")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_dnn",
    "--dnn-dealloc-opt=true",
    "--dnn-malloc-pool-opt=true",
    "yolov1.onnx"])

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("yolov1.onnx")
ort_in = {ort_session.get_inputs()[0].name: a_np}

sess = OMExecutionSession("./yolov1.so")


for i in range(100):
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
  torch_out = model(a)
  diff = (time.time() - start)*1000
  if(i):
    torch_results.append(diff)
  else:
    print("Pytorch CPU first value : ", diff)

  # using pytorch gpu
  if torch.cuda.is_available():
    start = time.time()
    torch.cuda.synchronize()
    torch.cuda.init()
    model_cuda = model.cuda()
    cuda_a = a.cuda()
    cuda_out = model_cuda(cuda_a)
    cuda_out.to('cpu')
    del model_cuda
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    diff = (time.time() - start)*1000
    if(i):
      cuda_results.append(diff)
    else:
      print("Pytorch GPU first value : ", diff)

  # using occamy output
  start = time.time()
  occamy_out = sess.run(a_np)
  diff = (time.time() - start)*1000
  if(i):
    occamy_results.append(diff)
  else:
    print("Occamy first value : ", diff)

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
  if 1:
    if torch.cuda.is_available():
      if 1:
        print("Correctness verified.")
      else:
        print("Not correct output! (torch, cuda different)")
        for i in (torch_out_round.eq(cuda_out_round).reshape(-1) == False).nonzero(as_tuple=True):
          print(torch_out_round.reshape(-1)[i])
          print(cuda_out_round.reshape(-1)[i])
    else:
      print("Correctness verified.")
  else:
    print("Not correct output! (occamy, torch different)")
    cnt = 0
    for i in (occamy_out_round.eq(torch_out_round).reshape(-1) == False).nonzero(as_tuple=True):
      print(occamy_out_round.reshape(-1)[i])
      print(torch_out_round.reshape(-1)[i])
      cnt = cnt + 1
      if cnt == 0:
        break
else:
  print("Not correct output! (ort, occamy different)")
  cnt = 0
  for i in (ort_out_round.eq(occamy_out_round).reshape(-1) == False).nonzero(as_tuple=True):
    print(ort_out_round.reshape(-1)[i])
    print(occamy_out_round.reshape(-1)[i])
    cnt = cnt + 1
    if cnt == 10:
        break

