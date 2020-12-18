# animeSR
Super-Resolution unsing deep neural networks for anime images and videos.

The datasets used for this project have been taken from imageboards based on [Danbooru Image Board framework](https://github.com/danbooru/danbooru) using a python script to download every 1080p images by emitting http requests to the imageboards' API.

The advantage of using a neural network for the upscaling task is that by viewing thousands of anime images, the neural network has learned anime drawings patterns and is able to reproduce details without making them blurry. By using other losses than the standard MSE like perception loss, the neural networks can also make better looking images and better qualitative results, even though the usual PSNR used for benchmarks might be lower.


## References
I list all the papers I have read for this project. Not every paper has been specifically reproduced in the code.


[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)  
[Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)  
[Improving the speed of neural networks on CPUs](https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/37631.pdf)  
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)  
[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)  
[Convolutional Neural Networks with Dynamic Regularization](https://arxiv.org/abs/1909.11862)  
[Densely Residual Laplacian Super-Resolution](https://arxiv.org/abs/1906.12021)  
[Deep Learning for Image Super-resolution: A Survey](https://arxiv.org/abs/1902.06068)  
[Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)  


