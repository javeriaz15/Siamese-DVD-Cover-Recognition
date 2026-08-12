# One-Shot DVD Cover Recognition with a Siamese Network
**Programmd by**: Juwairiah Zia (javeriazia15@gmail.com) and Enze cui (cuienze2015@gmail.com)<br />
**Data Set**: Stanford Mobile Visual Search Database<br />
**Project**:This project describes an algorithm for a mobile photo app where you take a picture of a DVD, and the app tells you all the information about it. The output should be to find out the movie that the DVD cover belongs to.<br />

## Problem
Each class of a DVD cover have a single training data instance 
## Solution
### Expanding dataset
Data augmentation was performed on each training image (per class) to expand the dataset
### Object detection and classification
- An image pair of DVD covers were used to trained two identical convolutional neural networks with the exact same weights
- Feature vector extracted as an output are then labeled as similar or dissimilar
### Why Siamese Neural Network
Siamese Neural Network learns to differentiate between two inputs instead of classifying them. It determines if the two inputs are different or similar, and the input can be anything. The advantages are less training data, less memory required, less computational cost, and time-consuming. Moreover, we don't need to worry about the number of classes.
