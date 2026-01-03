# Sign_Language_Detection_System

### Description :
Develop a system that detects American Sign Language (ASL) alphabets in real-time using a webcam, employing MediaPipe for hand landmark detection and OpenCV for video capture and display. Random Forest classifier model from scikit-learn is employed for alphabet recognition and containing the feature to hear, write, erase and dark theme. The model is then deployed using streamlit for the interactive user interface.

### Languages Used : 
1. MediaPipe
2. OpenCV
3. Streamlit


### Dataset from Kaggle :
GitHub Repository for Sign Language to Speech: Unvoiced [https://github.com/grassknoted/Unvoiced]

About
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.

Content
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images.
