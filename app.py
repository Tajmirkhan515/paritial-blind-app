from dotenv import load_dotenv
from PIL import Image
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except:
    from transformers import BlipProcessor, BlipForConditionalGeneration


from PIL import Image, UnidentifiedImageError
import requests

import torch
# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
import time

import os
import streamlit as st

import pandas as pd

from PIL import Image
import os
import datetime


import cv2
from PIL import Image
import pytesseract
import os
# Set the path to the Tesseract executable
# Update this path to point to your Tesseract installation if needed
pytesseract.pytesseract.tesseract_cmd = r'tesseract.exe'  # Assuming Tesseract is in the PATH

import openai
from gtts import gTTS





# Function to save the image
def save_image(image):
    # Ensure the directory exists
    directory = "captured_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create a unique filename based on the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/captured_image_{timestamp}.jpg"

    # Save the image
    with open(filename, "wb") as f:
        f.write(image.getvalue())

    return filename

def text_to_speech(text, output_file='output.mp3'):
    """
    Convert text to speech and save it as an MP3 file.

    Args:
    text (str): The text to convert to speech.
    output_file (str): The name of the output MP3 file. Default is 'output.mp3'.

    Returns:
    None
    """
    try:
        # Create a gTTS object with the given text and language
        tts = gTTS(text=text, lang='en')
        # Save the speech to the output file
        tts.save(output_file)
        print(f"Speech saved to {output_file}")
        # Optionally play the audio file (this works on Windows)
        os.startfile(output_file)
    except:
        print(" ")


def texGeneration(text):

    client = openai.OpenAI(
        api_key="647bab5949254a898c50e57bccda014a",
        base_url="https://api.aimlapi.com",
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Please provide a detailed and concise description of the following image that I took from a camera, and I am blind. Focus on describing the key elements, objects, and environment visible in the image. Do not include any extra information or explanations like inside image. Just provide a clear and accurate description of what is present in the enviroment. The content of in the enviroment is:"+text
        },
    ])
       

    message = response.choices[0].message.content
    print("message from server : ",message)
    text_to_speech(message)


# Streamlit UI
st.title("Camera Capture")
# Display camera input widget
photo = st.camera_input("Take a photo")
if photo:
    text=""
    with st.spinner("Processing your image..."):
        # Display the photo
        #st.image(photo, caption="Captured Image", use_column_width=True)

        # Save the image locally
        image_path = save_image(photo)
        #image_path="captured_images/photo4.jpg"
        st.write(f"Image saved locally as: {image_path}")
        try:
            image = Image.open(image_path)
        except:
            #print(f"Error: {e}")
            print(" ")
            exit()
       

        # Process the image and generate a caption
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs,
                            max_new_tokens=50)
        caption=" objects in image: "
        caption += processor.decode(out[0], skip_special_tokens=True)
        text +=caption
        print("Generated Caption:", caption)

        
        
            # Read the image using OpenCV
        img = cv2.imread(image_path)
        print("done 1")
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        no_noise_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(no_noise_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours were found
        if contours:
            # Sort contours and get the largest one
            cnt_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
            cnt = cnt_sorted[-1]
            
            # Get bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_image = no_noise_image[y:y+h, x:x+w]
            
            # Convert the image to a PIL Image object for OCR
            pil_image = Image.fromarray(cropped_image)
            
            # Perform OCR
            try:
                ocr_result=", text inside in image: "
                ocr_result += pytesseract.image_to_string(pil_image)
                text +=ocr_result
                # Check if OCR result is empty
                if ocr_result.strip():
                    print("Detected Text:")
                    print(ocr_result)

                else:
                    print(" ")
            except:
                print(" ")
        else:
            print("No contours found in the image.")
            text+=". there is no text inside in the image"
    print(" complete text ",text)
    texGeneration(text)
    st.success("Processing complete!")




