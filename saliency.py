import argparse
import os
import sys

import numpy as np
from scipy import misc
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16, vgg19
from torchvision.utils import save_image

from lib.gradients import VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image
from lib.labels import IMAGENET_LABELS

import glob
import time
import tarfile


def main():
    cuda = False
    n_samples = 10
    start = time.time()

    target_layer_names = ['35']
    target_index = None

    cwd = os.getcwd()
    
    # For temporary
    #datasetDir = "datasets"
    outputDir = "mnist_output"
    outputNpyPath = "mnist_test.npy"
    datasetPath = "mnist_1000_datasets.tar"
    tmpDir = "temp_data"
    tar_tmp = tarfile.open(datasetPath)
    tar_tmp.extractall(tmpDir)
    tar_tmp.close()

    if(os.path.exists(outputNpyPath)):
        os.remove(outputNpyPath)
    fileList = glob.glob(tmpDir + "/**/*.png", recursive=True)
    print("Total dataset size : " + str(len(fileList)))

    with open(outputNpyPath, 'wb') as f : 
        start = time.time()
        for filePath in fileList:
            img = cv2.imread(filePath, 1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            preprocessed_img = preprocess_image(img, cuda)
            model = vgg19(pretrained=True)
            output = model(preprocessed_img)
            pred_index = np.argmax(output.data.cpu().numpy())
            print('Prediction: {}'.format(IMAGENET_LABELS[pred_index]))

            smooth_grad = SmoothGrad(
                pretrained_model=model, cuda=cuda, n_samples=n_samples, magnitude=True)
            smooth_saliency = smooth_grad(preprocessed_img, index=target_index)
            output_path = os.path.join(cwd, outputDir, filePath)
            print("Output Path : " + output_path)
            save_as_gray_image(smooth_saliency, output_path)
            print('Saved smooth gradient image')
            np.save(f, smooth_saliency)

        timeConsumed = "Time cost : " + str(time.time() - start)
        print(timeConsumed)

if __name__ == '__main__':
    main()
