# KinshipVerification

Our main aim was to replicate the results that was achieved by the following [paper](https://arxiv.org/abs/2006.11739).

The verification is a four step process:

1) Redetecting and Realigning the faces from landmarks using the [Retina-Face](https://pypi.org/project/retina-face/).
2) Data Augmentation and Data transformation by mirrorring, adding random color, contrast and jitter to the image.
3) Extracting features from ResNet101 architecture as the backbone network which is trained with ArcFace loss using cleaned MS-Celeb-1M dataset. This complete model is called as [arcface_r100_v1](https://insightface.ai/arcface). This architecture is then trained on our train-processed directory. The output of the network is an embedding of 512 dimensions.
4) Compare two images using the embedding obtained from step 2 for the test images using Cosine distance as the metric and predict kin or non kin based on a tuned threshold.

**Step 1 :**

Use the preprocess_train_images.py to perform the first step. The script takes two arguments, first argument is the train_images directory and the second argument is the output directory where the [processed images](https://drive.google.com/drive/u/1/folders/1JlSeDMo9eaIMgMT4EErw3v4qecruiNZH) will be saved to.
Use the preprocess_test_images.py to do the same step for [test directory images](https://drive.google.com/drive/u/1/folders/1yfWpawfayqFM_DpTlqKIjetIzSDFq8Ee). There is no need to run these scripts as we have already saved the results in the google drive.

**Step 2 :**

In this step, Data augmentation is performed. augment_images.ipynb file is use to augment the training directory(output of step 1). The output of this step is [train-faces-augmented-1](https://drive.google.com/drive/u/1/folders/1yfWpawfayqFM_DpTlqKIjetIzSDFq8Ee).We produce new sets of images by mirroring the image and storing it to the google drive. The new sets of images are further transformed by adding random color and jitter to the image. Again there is no need to run these scrips as the results are stored.

**Step 3 :**

Step 1 and 2 is a time consuming step. As a result, We have uploaded the processed images of train and test directory to [google drive](https://drive.google.com/drive/u/1/folders/1gFTrdDxYnFQ_H7GyZIu3GVfmKOWdzpVD). The folders are train-faces-augmented-1 and test-processed respectively. To extract the features by training the model on the train directory as mentioned in step 2, simply run the Kinship Recognition Starter Notebook.ipynb file. Please make sure to adjust the root directory in the notebook. The training step is to classify the images in the train-faces-augmented-1 to a particular family.The intuition is that the network learns an embedding such that people belonging to a particular family are embbedded closer to each other. This step also outputs the model architecture and model weights obtained during the training process and is saved into my_train_models_norm directory. For example : 'export_arcface_families_ft-0060.params' is the weights of the trained model after 60 epochs and 'export_arcface_families_ft-symbol.json' is the network architecture.

**Step 4 :**

Run the predictions.ipynb file in google colab. This script builds the network by loading the network architecture file('export_arcface_families_ft-0060.params') as well as the weights file ('export_arcface_families_ft-symbol.json'). The forward pass on the test images generated an embedding of the image of 512 dimensions. Two images are compared using cosine distance as the metric. Finally we have kept the threshold as 0.1 to predict whether the images are kin or non kin based on the cosine metric.

**References :**

1) [Recognizing Families in the wild] (https://arxiv.org/abs/2002.06303)
2) [Achieving Better Kinship Recognition Through Better Baseline] (https://arxiv.org/pdf/2006.11739v1.pdf)
3) [Reference Implementation of Achieving Better Kinship Recognition Through Better Baseline] (https://github.com/vuvko/fitw2020)
