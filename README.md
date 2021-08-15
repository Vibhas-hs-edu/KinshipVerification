# KinshipVerification

Our main aim was to replicate the results that was achieved by the following [paper](https://arxiv.org/abs/2006.11739).

The verification is a three step process:

1) Redetecting and Realigning the faces from landmarks using the [Retina-Face](https://pypi.org/project/retina-face/).
2) Extracting features from ResNet101 architecture as the backbone network which is trained with ArcFace loss using cleaned MS-Celeb-1M dataset. This complete model is called as [arcface_r100_v1](https://insightface.ai/arcface). The output of this an embedding of 512 dimensions.
3) Compare two images using the embedding obtained from step 2 for the test images using Cosine distance as the metric and predict kin or non kin based on a tuned threshold.
