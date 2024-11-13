import streamlit as st
import tensorflow as tf
import numpy as np

#tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:/Users/hansi/OneDrive/Desktop/fruit_veg/trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single img to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # max element

    

#sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page',['Home','About Project','Prediction'])


#main page

if app_mode == 'Home':
    st.header('FRUIT & VEGETABLE RECOGNITION SYSTEM')
    image_path = "home_img.jpg"
    st.image(image_path)
    
    # Adding brief points about the project
    st.subheader("Project Highlights")
    st.markdown("""
    - **Goal**: Identify various fruits and vegetables using AI.
    - **Model**: Built with a Convolutional Neural Network (CNN) for image classification.
    - **Dataset**: Contains labeled images of over 20 fruits and vegetables.
    - **User Input**: Upload an image for instant prediction.
    - **Applications**: Potential use in health apps, diet tracking, and smart grocery checkout.
    """)
    
    
    
#About project
elif app_mode == 'About Project':
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("The dataset contains images of the following food items:")
    st.text("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.text("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category.")
    st.text("2. Test: Contains 10 images per category.")
    st.text("3. Validation: Contains 10 images per category.")
    
#prediction page

elif app_mode == "Prediction":
        st.header("Model Pediction")
        test_image = st.file_uploader("Choose an image")
        if st.button("Show Image"):
            st.image(test_image,width=4,use_container_width =True)
            
        #prediction button
        if st.button('Prediction'):
            st.balloons()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            #reading label 
            with open("labels.txt") as f:
                content = f.readlines()
            label = []
            for i in content:
                label.append(i[:-1])
            st.success("Model is predicting it's a {}".format(label[result_index]))
            


    