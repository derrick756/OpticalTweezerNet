# OpticalTweezerNet
Here we propose a residual neural network (ResNet) to characterize the size of optically trapped nanoparticles based on temporally undersampled position measurements acquired by a camera with limited detection bandwidth. Using the time series measurements of the position of a single particle in traps with both strong and weak stiffnesses, this work tests the performance of the ResNet in accurately determining particle size.

# Installation
Python 3.6 is recommended.


[python](https:www.python.org/).


### Install tensorflow


[tensorflow](https://www.tensorflow.org/).


### Install dependent packages
**1.Numpy**


pip install numpy


**2.Scipy**


pip install Scipy


**3.Matplotlib**


pip install Matplotlib


**4.Pandas**


pip install pandas


# Download the simulation code to generate training data
The simulation codes for generating theoretical constrained Brownian profiles can be downloaded on Google drive. The link is below


Download at: [Google drive](https://drive.google.com/drive/folders/1iSCFiVDbk9VcOxf4U-Y5a56myzIPHHSo).


**1.Training the model** 


Run the files "CNN-Tweezer.py" and "ResNet-Tweezer.py" to train the particle property prediction neural networks.


**2.Predict nanoparticles size**


Run the file "size.py" to predict the size.





