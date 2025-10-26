import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tempfile
import os
import pandas as pd

# Constants
IMAGE_SIZE = 256
CHANNELS = 3
MODEL_PATH = 'plant_disease_cnn_model.h5'
imga = 'Peer_Educators-removebg-preview.png'

if not os.path.exists('hamd.csv'):
    df = pd.DataFrame(columns=['Name', 'Number', 'Town', 'Age', 'Specialisation'])
    df.to_csv('hamd.csv', index=False)

def show_page(page_name):
    if page_name == "main_page":
        main_page()
    elif page_name == "search_page":
        search_page()
    elif page_name == "botanist_page":
        botanist_page()
    elif page_name == "docs_page":
        docs_page()
    elif page_name == "intro_page":
        intro_page()

def intro_page():
    st.image(imga, width = 500)
    st.write("*Welcome to Flora Scan, a comprehensive platform designed to facilitate plant disease detection and connect students with botanists for plant-related consultations. This application is built to address real-world problems through an interactive and user-friendly interface.*") 
    
    st.header("*Key Features*")
    st.write("1. *Plant Disease Detection*:")
    st.write("   - Upload an image or capture one using your webcam.")
    st.write("   - Get accurate predictions about plant diseases using a pre-trained deep learning model.")
    st.write("   - Text-to-speech functionality announces the predicted results for accessibility.")
    st.write("2. *Botanist Registration and Search*:")
    st.write("   - Botanists can register their details, including specialization and location, via a simple form.")
    st.write("   - Students can search for botanists based on geographical proximity and specialization.")
    st.write("3. *Data Persistence*:")
    st.write("   - All registered botanist data is stored in a CSV file (hamd.csv) for easy retrieval and updates.")

    st.header("*How It Works*")
    st.write("1. *For Students*:")
    st.write("   - Select the 'Student' option on the homepage.")
    st.write("   - Use the search feature to find botanists based on your city and preferences.")
    st.write("2. *For Botanists*:")
    st.write("   - Select the 'Botanist' option on the homepage.")
    st.write("   - Fill out the registration form with essential details.")
    st.write("   - Your profile will be available for students to find and connect with you.")
    st.write("3. *Plant Disease Detection*:")
    st.write("   - Upload or capture an image of a plant.")
    st.write("   - The system processes the image using a pre-trained deep learning model (model.h5) to predict the disease.")

    st.header("*Technologies Used*")
    st.write("- *Streamlit*: For building an interactive web interface.")
    st.write("- *TensorFlow*: For plant disease prediction using deep learning.")
    st.write("- *Pandas*: For managing and processing user and botanist data stored in hamd.csv.")
    st.write("- *Pillow*: For image handling and preprocessing.")
    st.write("- *pyttsx3*: For text-to-speech functionality to announce predictions.")

    st.header("*Conclusion*")
    st.write("Flora Scan bridges the gap between technology and practical problem-solving by offering tools for agriculture and plant-related consultation. Whether you're a farmer seeking plant disease solutions or a student looking for botanical guidance, this platform is here to help.")

    # Step-by-Step Guide for Students
    st.header("How to Use (For Farmers)")
    with st.expander("Step 1: Register"):
        st.write("""
        - On the homepage, select the 'Farmer' option.
        - Fill in the registration form with your details:
            - *Name*
            - *Age*
            - *City*
            - *Gender*
        - Submit the form to save your details in the system.
        """)

    with st.expander("Step 2: Search for Botanists"):
        st.write("""
        - Use the 'Search for Botanists' feature to find botanical professionals in your city.
        - Enter your city and view a list of botanists that match your location.
        - This list includes botanists' names, specialization, and contact details.
        """)

    with st.expander("Step 3: Connect with Botanists"):
        st.write("""
        - Select a botanist from the search results to view their complete profile.
        - Contact the botanist directly for consultation or further guidance.
        """)

    # Step-by-Step Guide for Botanists
    st.header("How to Use (For Botanists)")
    with st.expander("Step 1: Register"):
        st.write("""
        - On the homepage, select the 'Botanist' option.
        - Fill out the registration form with your details:
            - *Name*
            - *Contact Information*
            - *Age*
            - *Specialization*
            - *City of Practice*
        - Submit the form to save your profile and make it available for students to search.
        """)

    with st.expander("Step 2: Manage Your Profile"):
        st.write("""
        - Your profile will now be visible to students searching for botanists.
        - Ensure your details are accurate and up-to-date to improve connections with students.
        """)

    # Step-by-Step Guide for Plant Disease Detection
    st.header("How to Use (Plant Disease Detection)")
    with st.expander("Step 1: Capture or Upload an Image"):
        st.write("""
        - Go to the 'Plant Disease Detection' page.
        - Choose an option:
            - *Upload an Image:* Upload a clear image of the plant showing possible signs of disease.
            - *Capture an Image:* Use your device's camera to take a picture of the affected plant.
        """)

    with st.expander("Step 2: Analyze the Image"):
        st.write("""
        - The app will preprocess the image and use the AI model (model.h5) to detect and predict the plant disease.
        - Results will be displayed on the screen with the name of the disease and suggestions for further action.
        """)

    with st.expander("Step 3: Listen to the Result"):
        st.write("""
        - The app includes a text-to-speech feature.
        - It will announce the predicted plant disease, making it easier for users to understand the result.
        """)

    st.header("Tips for a Better Experience")
    st.write("""
    - For *Plant Disease Detection*:
        - Ensure the plant image is clear, well-lit, and focused.
        - Avoid blurry or obstructed images for better accuracy.
    - For *Students and Botanists*:
        - Keep your details updated for smoother interactions and better matches.
        - Use the app regularly to stay connected and informed.
    """)

    if st.button("Next"):
        st.session_state.page = "main_page"

def main_page():
    st.write("*This portal helps you find botanists.*")
    col1, col2 = st.columns([1, 2])

    farmer = st.checkbox("Farmer")
    botanist = st.checkbox("Botanist")

    st.session_state.farmer = farmer
    st.session_state.botanist = botanist

    if st.button("Next"):
        if st.session_state.farmer:
            st.session_state.page = "search_page"
        elif st.session_state.botanist:
            st.session_state.page = "botanist_page"

def search_page():
    st.title("Plant Disease Detection")

    # Load the pre-trained model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Upload the correct file.")
        st.stop()

    loaded_model = load_model(MODEL_PATH)

    # Initialize ImageDataGenerator to get class names
    try:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(
            'C:/Users/LOQ/Desktop/Python/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=32,
            class_mode='sparse'
        )
        class_names = list(train_generator.class_indices.keys())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Sidebar for user actions
    action = st.sidebar.radio("Select an action:", ("Capture from Webcam", "Upload an Image"))

    def preprocess_image(image):
        """Preprocess the image for prediction."""
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(image):
        """Predict the class and display the result."""
        prediction = loaded_model.predict(image)
        index = np.argmax(prediction)
        predicted_class = class_names[index]
        st.success(f"Prediction: {predicted_class}")

    # Handle actions
    if action == "Capture from Webcam":
        st.write("*Webcam Capture*")
        captured_image = st.camera_input("Capture an image")
        if captured_image is not None:
            try:
                # Use temporary file for captured image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(captured_image.getvalue())
                    temp_file_path = temp_file.name

                # Load, display, and preprocess the captured image
                img = Image.open(temp_file_path)
                st.image(img, caption="Captured Image", use_column_width=True)
                img_array = preprocess_image(img)

                # Predict
                predict(img_array)

                # Cleanup temporary file
                os.remove(temp_file_path)

            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")

    elif action == "Upload an Image":
        st.write("*Image Upload*")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Load, display, and preprocess the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                img_array = preprocess_image(img)

                # Predict
                predict(img_array)

            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")

    if st.button("Search"):
        st.session_state.page = "docs_page"

def botanist_page():
    st.title("*Botanist Registration Form*")
    with st.form(key='botanist_form'):
        st.write("Please enter your details:")

        name = st.text_input("Enter your name:")
        number = st.text_input("Enter your number:")
        town = st.text_input("Enter your city")
        age = st.number_input("Enter Your Age", min_value=1, max_value=120)
        specialisation = st.text_input("Enter Your Specialisation")
        email = st.text_input("Enter Your E-Mail")

        submitted = st.form_submit_button("Submit")

        if submitted:
            data = {
                'Name': name,
                'Number': number,
                'Town': town,
                'Age': age,
                'E-Mail': email,
                'Specialisation': specialisation,
            }
            new_data_df = pd.DataFrame([data])
            df = pd.read_csv('hamd.csv')
            df = pd.concat([df, new_data_df], ignore_index=True)
            df.to_csv('hamd.csv', index=False)
            st.success("Botanist registration successful!")

    if st.button("Back"):
        st.session_state.page = "main_page"

def docs_page():
    st.title("*Search a Botanist*")

    # User input fields
    student_name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=1, max_value=120)
    city = st.text_input("Enter your city:")
    gender = st.radio("Select your gender:", ("Male", "Female"))

    # Save user input into session state (optional)
    st.session_state.student_name = student_name
    st.session_state.age = age
    st.session_state.city = city
    st.session_state.gender = gender

    # Search button functionality
    if st.button("Search"):
        if city:
            # Load the data
            df = pd.read_csv('hamd.csv')

            # Filter by city, handling NaN values in 'Town' column
            filtered_df = df[df['Town'].str.contains(city, case=False, na=False)]

            # Display the results
            if not filtered_df.empty:
                st.write("Botanists in your town:")
                st.dataframe(filtered_df)
            else:  
                st.write("No botanists found matching your criteria.")

    if st.button("Back"):
        st.session_state.page = "main_page"

if 'page' not in st.session_state:
    st.session_state.page = "intro_page"

show_page(st.session_state.page)