# The SimNets Architecture's Implementation in Caffe

This is the official implementation of the SimNets Architecture, as featured in the articles:
* [SimNets: A Generalization of Convolutional Networks](https://arxiv.org/abs/1410.0781). Nadav Cohen and Amnon Shashua. NIPS 2014 Deep Learning Workshop.
* [Deep SimNets](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Cohen_Deep_SimNets_CVPR_2016_paper.html). Nadav Cohen, Or Sharir and Amnon Shashua. CVPR 2016.
* [Tensorial Mixture Models](https://arxiv.org/abs/1610.04167). Or Sharir, Ronen Tamari, Nadav Cohen and Amnon Shashua. arXiv preprint. October, 2016.

Our implementation is based on the Caffe framework. If you have already installed Caffe, then it is typically enough to just copy over your Makefile.config to this project, and run `make all`. The CMake build option has been temporarily removed from our fork because of incompatibilities. You can find examples on how to use the SimNets architecture in the [experiments repository of our new article: "Tensorial Mixture Models"](https://github.com/HUJI-Deep/TMM).

The rest of the README is taken from the original repository.

---

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
