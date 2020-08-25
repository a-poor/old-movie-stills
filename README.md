# Analizing Movie Stills with Machine Learning

_created by Austin Poor_

My goal is to analyze and explore movies based on their imagery.

I collected the data by scraping film stills from the site [film-grab.com](https://film-grab.com/) and stored them on Amazon S3. I collected `132,617` stills corresponding to `2,147` movies.

I then used the Python package `PIL` to standardize all of the images to be `.png` files and to have the dimensions `224px x 126px` (which means an aspect ratio of `16:9`). The reason being, the VGG-16 model takes an input of `224 x 224` and, after inspecting the images, the aspect ratio of `16:9` seemed to be one of the most common in HD video while still being permissive enough to include as much image detail as possible (compared to a possibly _more_ common cinematic aspect ratio like `2.39:1` which would require cropping out a lot of detail from many film stills). See the notebook [resize-images.ipynb](./resize-images.ipynb).

Additionally, some of the source images were already `.png` files with alpha channels. Since VGG-16 expects inputs to only have red, green, and blue channels (plus the fact that our _current_ common idea of movies doesn't use any transparency) I threw out the alpha channel for those images. See the notebook [fix-image-dimensions.ipynb](./fix-image-dimensions.ipynb).

In order to convert the raw images to embedding vectors, I used a pretrained VGG-16 model, [from PyTorch](https://pytorch.org/docs/stable/torchvision/models.html), without the final layer of the classifier, which outputs a vector of length `4,096`. See the notebook [image-to-vgg16.ipynb](./image-to-vgg16.ipynb).

Next, I converted the `4,096`-dimensional embeddings to `2`-dimensional embeddings using `Principal Component Analysis`, which would allow me to plot the relationship between films. See the notebook [vgg16-to-pca.ipynb](./vgg16-to-pca.ipynb)

The result of the PCA transform will be a gaussian blob of image points but I want to place the images into a grid so they can be visualized. In order to make that transformation, I used the [`RasterFairy`](https://github.com/Quasimondo/RasterFairy) Python library. See the notebook [pca-to-grid.ipynb](./pca-to-grid.ipynb).


*** 

## Notes

### Ideas to Explore
* Clustering similar film stills
* Comparing movies as sequences of stills
* Recommending movies with similar looks
* Plotting stills in 2D grid space
* Using HSL rather than RGB
* Creating color palettes from movies

### Vectorizing Stills
* PCA
* VGG16
* Conv-AutoEncoder
* tSNE (for 2D grid)

