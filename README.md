# Fruits-and-Vegetables-recognition
An AI-based image classification system that recognizes various fruits and vegetables using a Convolutional Neural Network (CNN). This project leverages deep learning to make accurate predictions on 20+ types of fruits and vegetables.

# Demo
Upload an image of a fruit or vegetable, and the model will predict the type instantly.

# Features
-Real-Time Prediction: Upload an image to get an instant prediction of the fruit or vegetable type.

-Deep Learning Model: A CNN model, trained on a curated dataset of fruits and vegetables, ensures high accuracy.

-Extensive Dataset: Includes over 10 categories of fruits and 10 categories of vegetables.

-Interactive UI: Built with Streamlit for an easy-to-use interface.

# Project Structure
-main.py: Main file for running the Streamlit application.

-trained_model.h5: Pre-trained CNN model file.

-labels.txt: Text file with labels for each fruit and vegetable type.

# Sample Images
Here are some sample images from the dataset: 

(Apple, Banana, Grapes, Mango, Orange, Carrot, Tomato, Potato, Cucumber, Capsicum)

# How It Works
The dataset contains:

-Training Set: 100 images per category.

-Testing Set: 10 images per category.

-Validation Set: 10 images per category.

# Use Cases
-Smart Grocery Systems: Automate produce identification for checkout.

-Health Apps: Recognize food items to assist in diet tracking.

-Retail: Automate produce recognition for checkouts in grocery stores.

-Education: Useful for learning about CNNs and image classification.

# Technologies Used
-TensorFlow: For building and training the CNN model.

-Streamlit: For creating the web-based user interface.

-NumPy: For handling image data as arrays.

# Challenges Faced and Solutions
1. Data Collection and Preparation
   
   -Challenge: Finding a diverse dataset with a wide variety of fruits and vegetables, each with enough images for training, validation, and testing.
   
   -Solution: Curated a dataset with 100+ images per category and split it into train, test, and validation sets. Applied data augmentation techniques
   
   (e.g., rotation, flipping) to increase dataset diversity.

3. Handling Model Overfitting
   
   -Challenge: The model initially showed signs of overfitting, performing well on the training data but poorly on unseen images.
   
   -Solution: Implemented regularization techniques such as dropout layers and used data augmentation. Also, fine-tuned hyperparameters to achieve a balanced performance.
   
5. Optimizing Training Time
   
    -Challenge: Training a CNN on a large dataset with limited computational resources resulted in long training times.
   
    -Solution: Optimized the model’s architecture to reduce complexity without losing accuracy. Used transfer learning with a pre-trained model as the base, reducing both 

     the training time and resource requirements.

7. Image Size and Quality Standardization
   
   -Challenge: Images varied in resolution, causing inconsistencies in model input.
   
   -Solution: Resized all images to a standard size (64x64 pixels) to maintain consistency, balancing between image quality and processing speed.

9. Ensuring High Prediction Accuracy
    
   -Challenge: Some fruit and vegetable categories looked similar (e.g., different types of peppers), leading to misclassification.
   
   -Solution: Improved data quality by refining the dataset with high-quality images and trained the model longer, monitoring both training and validation accuracy.

    Additionally, tweaked the CNN architecture to enhance accuracy on complex or similar-looking items.

11. Streamlit Integration for User Interface
    
    -Challenge: Integrating the trained model with Streamlit and ensuring smooth functionality for uploading and predicting images.
    
    -Solution: Developed a streamlined workflow that allows users to upload images and instantly get predictions. Addressed issues with file handling and ensured model 

     compatibility with Streamlit.


   # HOME page Image of the Fruit and vegetables Prediction system
   <img width="955" alt="Screenshot 2024-11-12 235323" src="https://github.com/user-attachments/assets/57ad2e8d-ebda-49b5-a8f6-2424da19caed">

   # About Project webpage
   <img width="957" alt="Screenshot 2024-11-12 235339" src="https://github.com/user-attachments/assets/5320aa28-a303-4272-9b39-08f64e2da541">

   # Model Pridiction
   <img width="954" alt="Screenshot 2024-11-12 235354" src="https://github.com/user-attachments/assets/e8037908-bf07-4f49-945e-4c6793cc1c79">

   # Uploading vegetable image
   <img width="943" alt="Screenshot 2024-11-12 235448" src="https://github.com/user-attachments/assets/bbbb3cec-f39e-4dd4-af81-240fdf59f971">
   <img width="933" alt="Screenshot 2024-11-12 235501" src="https://github.com/user-attachments/assets/51f6c4dd-999c-4aac-bc67-10ff46a3885a">

   # Model is predicting it's a potato
   <img width="931" alt="Screenshot 2024-11-12 235515" src="https://github.com/user-attachments/assets/c9214bc6-b607-4d59-bd53-820b30340f21">

   # Uploading Fruit image
   <img width="941" alt="Screenshot 2024-11-12 235602" src="https://github.com/user-attachments/assets/2473a22d-aed2-47c1-9cee-afa73f42ab5f">

   # Model is predicting it's a banana
   <img width="938" alt="Screenshot 2024-11-12 235718" src="https://github.com/user-attachments/assets/36d1154c-aa61-433d-b377-c5faad8241a1">
   








   
