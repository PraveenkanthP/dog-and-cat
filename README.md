# Cat and Dog Image Classifier

This project aims to develop an image classification model to distinguish between images of cats and dogs using data science techniques in Python.

## Dataset

We used the Kaggle "Dogs vs. Cats" dataset, which contains thousands of images of cats and dogs. The dataset is organized into a directory structure with separate folders for the training and testing sets, each containing subfolders for cats and dogs.

If you'd like to use your own dataset, make sure it follows a similar directory structure:


dataset/
│
├── training_set/
│   ├── cats/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog1.jpg
│       ├── dog2.jpg
│       └── ...
│
└── test_set/
    ├── cats/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    └── dogs/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...


## Installation

1. Clone this repository:

   git clone https://github.com/yourusername/cat-dog-classifier.git
  
2. Install the required dependencies:

   pip install -r requirements.txt
   

## Usage

1. Navigate to the project directory:

   cd cat-dog-classifier
   

2. Run the training script to train the model:
 
   python train_model.py
  

3. After training, you can evaluate the model on the test set:

   python evaluate_model.py


## Model

We used a pre-trained VGG16 model as a base and fine-tuned it for our image classification task. The model achieved an accuracy of X% on the test set.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
