## ArSLR-ML : A Python-Based Machine Learning Application for Arabic Sign Language Recognition
The ArSLR-ML is a real-time interactive application that uses multi-class Support Vector Machines (SVM) machine learning applied in the classification procedure and MediaPipe in the feature extraction procedure to recognize static Arabic sign language gestures, focusing on numbers and letters and translating them into text and Arabic audio output. The ArSLR-ML was built within the PyCharm IDE using Python with a graphical user interface (GUI), thereby allowing for effective recognition of gestures. The application utilizes the laptop camera and GUI to capture hand gestures to create dataset for machine learning models and implement them in real time.


### ArSLR-ML workflow
The process of predicting Arabic Sign Language Recognition (ArSLR) using a multi-class Machine Learning-based model, specifically **support vector machines (SVM)**, involves six stages in total: Image Collection (stage 1), Feature Extract  ion (stage 2), Preprocessing (stage 3), Classification (stage 4), Model Evaluation (stage 5), and Finally Real-Time Recognition (stage 6). The architecture of the ArSLR model is depicted in Figure below:

![ArSLR model](https://github.com/user-attachments/assets/b546df32-2ab4-4347-898f-e5b2d18a302b)

### Related Article
This project is related to the research article titled "Static Arabic Sign Language Recognition in Real Time Using Machine Learning and MediaPipe" in 2024 1st International Conference on Emerging Technologies for Dependable Internet of Things (ICETI), IEEE, 2024, pp. 1–8. You can access the article through the following DOI link: 10.1109/ICETI63946.2024.10777193.

### Project Structure
The ArSLR-ML application consists of four main scripts; each script contains code to design the application and several functions to facilitate the design of the GUI and the execution of tasks:
GUI/: Scripts directory.
1. Main.py is responsible for the overall application workflow.
2. data_collection_widget.py, responsible for hand gesture image collection and preparation, and the functions in this script are:
   * opening: Function to open the sub-window that displays Arabic sign language gestures image.
   * open_data_collection: Function to use to collect images for each class.
   * open_collect_images_in_specific_folder: Function to modify the image in any class
   * create_dataset: Function to create a dataset from the image after processing in function "process_image".
   * process_image: Function to process all the images in classes.

3. training_Widget.py, responsible for classification model training and evaluation of the trained model, and the functions in this script are:
   * open_dataset_directory: This function opens the directory and ensures that the dataset file is found.
   * opening: Function to open the sub-window that displays Arabic sign language gestures image.
   * train_model: Function to train the dataset using SVM.
   * classification_report: Function to display classification report and confusion matrix.
4. testing_Widget.py, responsible for testing the trained model in real-time, and the functions in this script are:
   * open_dataset_directory: Function to open the directory and ensure finding the dataset file
   * convert_to_arabic_numerals: This function converts numbers to a string and replaces each digit with the corresponding Arabic numeral.
   * update_frame: This function displays the frame containing the live image of the camera for the estimation display.
   * draw_text_with_pil: Function to draw text with PIL for Arabic support.

### Requirements & Libraries
* Python v3.11 
* MediaPipe
* Scikit-learn
* Scikit-optimize
* OpenCV
* NumPy
* PyQt5

### Launch the ArSLR-ML App.
You can start the project by running the Main.py file in the root directory. This loads the application settings. The system starts by going to the Arabic Sign Language Recognition Model and then deciding whether we want to create a new dataset and train it using the SVM model and evaluate it or move to the real-time testing phase. It enables real-time recognition of static ArSL signs captured by a camera, converts the recognized signs into Arabic speech, and instantly displays them as text on the laptop screen.

