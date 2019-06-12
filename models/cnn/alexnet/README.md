# AlexNet
Alexnet is the winner of ILSVRC 2012.

# Paper
This model was published in NIPS 2012.

Link: [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

# Implementations

## AlexNet from pytorch/vision
- Code: [alexnet-torchvision.py](alexnet-torchvision.py)
- Note:
    - This model uses `nn.AdaptiveAvgPool2d` to allow the model to process images with arbitrary image size. [PR #746]
    - This model doesn't use Local Response Normalization as described in the original paper.
        - This model is implemented in Jan 2017 with pretrained model.
        - PyTorch's Local Response Normalization layer is implemented in Jan 2018. [PR #4667]
- References:
    - Model: [pytorch/vision](https://github.com/pytorch/vision/blob/ac2e995a4352267f65e7cc6d354bde683a4fb402/torchvision/models/alexnet.py)
    - PR #746: [pytorch/vision/pull/746](https://github.com/pytorch/vision/pull/746)
    - PR #4667: [pytorch/vision/pull/4667](https://github.com/pytorch/pytorch/pull/4667)