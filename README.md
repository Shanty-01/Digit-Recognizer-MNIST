# HandwrittenDigits-Recognizer
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shanty-01/HandwrittenDigits-Recognizer/blob/main/HandwrittenDigits_Recognizer.ipynb)

This project aims to evaluate and explain different benchmark models on MNIST dataset to determine the best approach for handwritten digit classification.
## About
* The models Support-vector machine (SVM) and K-Nearest Neighbors (KNN) are chosen for their strong baseline performance in classification tasks.
* And while Convolutional neural network (CNN) is more suited to image data dan simple multi-layer perceptron (MLP), an MLP model is also evaluated to see the extent of its capability in image classification.
* Evaluation is based on the training time and accuracy.
  
The dataset used is the Modified National Institute of Standards and Technology (MNIST) dataset, which consists of publicly available images of handwritten digits. In this project, the dataset is accessed from Google Colab's sample data folder:

  * `/content/sample_data/mnist_test.csv`

  * `/content/sample_data/mnist_train_small.csv`

The best-performing model is evaluated on newly photographed handwritten digits.

## Result
| Model     | Test Accuracy | Time(s) |
|------------|-----------|--------|
| Support Vector Machines(SVM)| 0.97   | 50.75   |
| K-nearest neighboor (KNN)    | 0.96      | **0.10**   |
| Multi-layer Perceptron (MLP)    | 0.96      | 661.30   |
| Convolutional Neural Network (CNN)   |**0.98**      | 718.74   |

* CNN gives the best accuracy even though it is the slowest to train, taking 718.74 seconds. This model is suitable when high accuracy is prioritized and longer training time is acceptable.
* KNN has the fastest time to train/create a model ready for prediction(0.10 seconds) but achieves slightly lower accuracy (0.96). This makes it ideal for scenarios where training time is important, and a slight trade-off in accuracy is acceptable.
* SVM achieves high accuracy (0.97) with a much shorter training time (50.75 seconds) compared to CNN. This model gives good compromise between accuracy and training time.
* CNN model with the highest testing accuracy struggles to predict digits with skewed positioning (translation) and thin-lined digit 1.

## To-Do Next
We could make the CNN model more robust and potentially achieve higher accuracy by :  

*   Data Augmentation  
    Handwritten digits retrieved from various sources in the real world may have skewed positions, various stroke widths or even distortions in the image. We could help our model generalize better to these images by applying distortion, translation and filters to the existing MNIST dataset and add them in our training dataset.
*   Hyperparameter Seach  
    Different choices of hyperparameters such as batch size, kernel size, learning rate,etc., could make a significant impact on the model's performance. We could explore the hyperparameter space using hyperparameter optimization algorithms such as grid search, random search, or Bayesian optimization to find the best configuration.
*   Model Architecture Exploration  
    Exploring alternative CNN architectures, such as deeper networks, residual connections, or attention mechanisms, might improve accuracy.

