## LoAd Network: Adaptive Deep Learning through Visual Domain Localization

Code for the paper:
[Adaptive Deep Learning through Visual Domain Localization](https://arxiv.org/ "Arxiv")\
Gabriele Angeletti, Barbara Caputo, Tatiana Tommasi

##### Note:
The torch version in `load_network_torch` is working, while the pytorch version in `load_network_pytorch` is still a work in progress.

### Abstract:
An open challenge in robot visual object recognition is the ability to generalize across different visual domains. A commercial robot, trained by its manufacturer to recognize a predefined number and type of objects, might be used in many settings, that will in general differ in their illumination conditions, type and degree of clutter, and so on. Recent works in the computer vision community deal with the generalization issue through domain adaptation methods, assuming as source the visual domain where the system is trained and as target the domain of deployment. All approaches assume to have access to images from all classes of interest in the target domain during training, an unrealistic condition in robotics applications. We address this issue proposing an algorithm for domain adaptation that takes into account the specific needs of robot vision. Our intuition is that the nature of the domain shift experienced mostly in robotics is local. We exploit this through the learning of maps that spatially ground the domain and quantify the degree of the domain shift among images, embedded into an end-to-end deep domain adaptation architecture. By explicitly localizing the roots of the domain shift we significantly reduce the number of parameters of the architecture to tune, we gain the flexibility necessary to deal with subset of categories in the target domain at training time, and we provide a clear feedback on the rationale behind any classification decision, which can be exploited in human-robot interactions. Experiments on two different settings of the iCub World database confirm the suitability of our method for robot vision, compared to existing state of the art approaches.

![Domain localization network](https://i.imgur.com/b5wJbeN.png)

![LoAd network](https://i.imgur.com/sKDqDFu.png)

### Usage
Todo

### License:
MIT

#### 3rd-party:
* [Grad-cam](https://github.com/ramprs/grad-cam)