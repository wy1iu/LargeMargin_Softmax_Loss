# Souce code for "Large-Margin Softmax Loss for Convolutional Neural Networks"

### Citation
The paper is available at https://arxiv.org/abs/1612.02295.

If the code helps your research, please cite our work.

    Large-Margin Softmax Loss for Convolutional Neural Networks
    Weiyang Liu, Yandong Wen, Zhiding Yu and Meng Yang
    Proceedings of The 33rd International Conference on Machine Learning. 2016: 507-516.

    @inproceedings{liu2016large,
      title={Large-Margin Softmax Loss for Convolutional Neural Networks},
      author={Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Yang, Meng},
      booktitle={Proceedings of The 33rd International Conference on Machine Learning},
      pages={507--516},
      year={2016}
    }

### Update
- 2017/5/25 Try to replace ReLU with PReLU, if you can not adjust lambda to make the network converge. (I also mentioned this in the note.)
- 2017/1/23 Fix a bug that lambda_min may change during backprop. Thanks [luoyetx](https://github.com/luoyetx)
- 2017/1/23 A mxnet implementation is also available at [here](https://github.com/luoyetx/mx-lsoftmax). Credit goes to [luoyetx](https://github.com/luoyetx).

### Files
- Caffe library
- L-Softmax Loss
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/largemargin_inner_prodcut_layer.hpp
  * src/caffe/layers/largemargin_inner_prodcut_layer.cpp
  * src/caffe/layers/largemargin_inner_prodcut_layer.cu
- mnist example
  * myexamples/mnist/mnist_test_lmdb
  * myexamples/mnist/mnist_test_lmdb
  * myexamples/mnist/model/mnist_train_test.prototxt
  * myexamples/mnist/mnist_solver.prototxt
- cifar10 example
  * myexamples/cifar10/model/cifar_train_test.prototxt
  * myexamples/cifar10/cifar_solver.prototxt
- cifar10+ example
  * myexamples/cifar10+/model/cifar_train_test.prototxt
  * myexamples/cifar10+/cifar_solver.prototxt

### Usage
- The prototxt of LargeMarginInnerProduct layer is as follows:

        layer {
          name: "ip2"
          type: "LargeMarginInnerProduct"
          bottom: "ip1"
          bottom: "label"
          top: "ip2"
          top: "lambda"
          param {
            name: "ip2"
            lr_mult: 1
          }
          largemargin_inner_product_param {
            num_output: 10 //number of outputs
            type: QUADRUPLE //value of m
            //only SINGLE (m=1), DOUBLE (m=2), TRIPLE (m=3) and QUADRUPLE (m=4) are available.
            base: 1000
            gamma: 0.000025
            power: 35
            iteration: 0
            lambda_min: 0
            //base, gamma, power and lambda_min are parameters of exponential lambda descent
            weight_filler {
              type: "msra"
            }
          }
          include {
            phase: TRAIN
          }
        }

- For specific examples, please refer to myexamples/mnist folder.

### Notes
- L-Softmax loss is the combination of "LargeMarginInnerProduct" layer and "SoftmaxWithLoss" layer.
- If the type of the layer is SINGLE/DOUBLE/TRIPLE/QUADRUPLE, then m is set as 1/2/3/4 respectively.
- mnist example can be run directly after compilation. cifar10 and cifar10+ requires datasets to be downloaded first.
- base, gamma, power and lambda_min are parameters for exponential lambda descent. lambda represents the approximation level to the proposed L-Softmax loss (refer to the experimental details in the ICML'16 paper). lambda will be decreased by the equation: lambda = max(lambda_min,base\*(1+gamma\*iteration)^(-power)). It is strong recommended that the user visualizes the lambda descent function before using the loss. The parameter selection is very flexible. Typically, when the optimization is finished, lambda should a sufficiently small value. Also note that, lambda is not always necessary. For MNIST dataset, the L-Softmax loss can work perfectly without lambda. Setting base to 0 can remove the lambda.
- lambda_min can vary according to the difficulty of datasets. For easy datasets such as mnist and cifar10, lambda_min can be zero. For large and difficult datasets, you should first try to set lambda_min as 5 or 10. There is no specific rule to set lambda_min, but generally, it should be as small as possible.
- Both ReLU and PReLU work well with L-Softmax loss. Empirically, PReLU helps L-Softmax converge easier.
- Batch normalization could help the L-Softmax network converge much easier. It is strong recommended to use it.

### Disclaimer
- This code is for research purpose only.

### Contact
If you have any questions, feel free to contact:
- Weiyang Liu (wyliu@gatech.edu)
- Yandong Wen (yandongw@andrew.cmu.edu)


### License
Copyright(c) all authors
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

