# KinshipVerification

Our main aim was to replicate the results that was achieved by the following [paper](https://arxiv.org/abs/2006.11739).

The verification is a three step process:

1) Redetecting and Realigning the faces from landmarks using the [Retina-Face](https://pypi.org/project/retina-face/).
2) Extracting features from ResNet101 architecture as the backbone network which is trained with ArcFace loss using cleaned MS-Celeb-1M dataset. This complete model is called as [arcface_r100_v1](https://insightface.ai/arcface). This architecture is then trained on our train-processed directory. The output of this is an embedding of 512 dimensions.
3) Compare two images using the embedding obtained from step 2 for the test images using Cosine distance as the metric and predict kin or non kin based on a tuned threshold.

Step 1 :

Use the preprocess_train_images.py to perform the first step. The script takes two arguments, first argument is the train_images directory and the second argument is the output directory where the processed images will be saved to.
Use the preprocess_test_images.py to do the same step for test directory images.

Step 2 :

Step 1 is a time consuming step. As a result, We have uploaded the processed images of train and test directory to [google drive](https://drive.google.com/drive/u/1/folders/1gFTrdDxYnFQ_H7GyZIu3GVfmKOWdzpVD). The folders are train-faces-processed and test-processed respectively. To extract the features by training the model on the train directory as mentioned in step 2, simply run the Kinship Recognition Starter Notebook.ipynb file. Please make sure to adjust the root directory in the notebook. This step also outputs the model architecture and model weights obtained during the training process and is saved into my_train_models_norm directory. For example : 'export_arcface_families_ft-0060.params' is the weights of the trained model after 60 epochs and 'export_arcface_families_ft-symbol.json' is the network architecture.
