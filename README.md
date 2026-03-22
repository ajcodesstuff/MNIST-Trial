This project was a test run to try and understand how basic computer vision works. The Training.py saves the data, and trains the model on it. After the training, it saves the model to the mnist_cnn.pth, and the saved model could be loaded for further training, or in the App.py, where it can be tested on the test data manually, with visual representation, or with user-made handwritten digits.

To run the code, you'd need the following libraries: torch, torchvision, customtkinter, pillow.

The estimated accuracy is ~98%, because it was trained on modified data to increase real-world applications. It fares okay-ish on the User-input, getting most of them right, but sometimes gets confused if the digit is a bit weird.

Overall, I am happy with how this turned out. Peace✌️
