# Person Re-identification Research Based on Deep Learning
This project codes are based on this github link. https://github.com/Cysu/open-reid

Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results.

## Installation

Install [PyTorch](http://pytorch.org/) (version >= 0.2.0). Although we support
both python2 and python3, we recommend python3 for better performance.

```shell
git clone https://github.com/Cysu/open-reid.git
cd open-reid
python setup.py install
```

## Examples

```shell
python examples/softmax_loss.py -d viper -b 64 -j 2 -a resnet50 --logs-dir logs/softmax-loss/viper-resnet50
```

This is just a quick example. VIPeR dataset may not be large enough to train a deep neural network.

Check about more [examples](https://cysu.github.io/open-reid/examples/training_id.html)
and [benchmarks](https://cysu.github.io/open-reid/examples/benchmarks.html).

## Relevant pictures
### Results example
![image](https://github.com/Tianyu97/Person-Re-identification-Research-Based-on-Deep-Learning/blob/master/images/results_example.png)
### Framework of the model
![image](https://github.com/Tianyu97/Person-Re-identification-Research-Based-on-Deep-Learning/blob/master/images/framework.png)
### DeeperCut keypoint detection
![image](https://github.com/Tianyu97/Person-Re-identification-Research-Based-on-Deep-Learning/blob/master/images/DeeperCut_keypoint_detection.jpg)
### Divide the parts of one person
![image](https://github.com/Tianyu97/Person-Re-identification-Research-Based-on-Deep-Learning/blob/master/images/divide_parts.png)
