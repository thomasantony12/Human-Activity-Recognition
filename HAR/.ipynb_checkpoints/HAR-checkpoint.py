import streamlit as st
import os
import cv2
import numpy as np
import pafy
import tensorflow as tf
from moviepy.editor import VideoFileClip
from pytube import YouTube
from keras.models import load_model
from keras.layers import ConvLSTM2D
from keras.initializers import Orthogonal

# Define the custom initializer
custom_initializer = Orthogonal(gain=1.0, seed=None)

# Define the path to your model file
model_path = 'D:\\Major project\\main\\my_model.keras'

# Load the model, specifying the custom_objects with the custom initializer
loaded_model = load_model(model_path, custom_objects={'Orthogonal': custom_initializer}, compile=False)

SEQUENCE_LENGTH = 20

def predict_single_action(video_file_path, SEQUENCE_LENGTH):  #
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    SEQUENCE_LENGTH = 20
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        IMAGE_HEIGHT = 224
        IMAGE_WIDTH = 224
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = loaded_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()


def download_youtube_videos(youtube_video_url, output_directory):
    '''
    This function downloads the youtube video whose URL is passed to it as an argument.
    Args:
        youtube_video_url: URL of the video that is required to be downloaded.
        output_directory: The directory path to which the video needs to be stored after downloading.
    Returns:
        title: The title of the downloaded youtube video.
    '''
    try:
        yt = YouTube(youtube_video_url)
        title = yt.title
        stream = yt.streams.get_highest_resolution()
        stream.download(output_directory)
        return title
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None

    # Check if the provided YouTube video URL is valid
    if 'youtube.com' not in youtube_video_url:
        raise ValueError("Invalid YouTube video URL. Please provide a valid YouTube video URL.")

    # Extract the video ID from the YouTube video URL
    video_id = youtube_video_url.split('v=')[-1]

    # Construct the complete YouTube video URL based on the video ID
    complete_video_url = f'https://www.youtube.com/watch?v={video_id}'

    # Create a video object which contains useful information about the video
    video = pafy.new(complete_video_url)

    # Retrieve the title of the video
    title = video.title

    # Get the best available quality object for the video
    video_best = video.getbest()

    # Construct the output file path
    output_file_path = f'{output_directory}/{title}.mp4'

    # Download the YouTube video at the best available quality and store it to the constructed path
    video_best.download(filepath=output_file_path, quiet=True)

    # Return the video title
    return title


st.title("Human Activity Recognition")
def process_input(videolink):
    # Process the input text here
    st.write("Input Text:", videolink)
videolink=st.text_input("Enter the youtube video link")
if st.button("Recognize"):
    process_input(videolink)

    # Make the Output directory if it does not exist
    test_videos_directory = 'D:\\Major project\\main\\test_videos'
    os.makedirs(test_videos_directory, exist_ok = True)

    # Download a YouTube Video.
    video_title = download_youtube_videos(videolink, test_videos_directory)
    st.write("TEST", video_title)
    # Get the YouTube Video's path we just downloaded.
    input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

    # Perform Single Prediction on the Test Video.
    predict_single_action(input_video_file_path, SEQUENCE_LENGTH) #, SEQUENCE_LENGTH

    # Display the input video.
    VideoFileClip(input_video_file_path, audio=False, target_resolution=(300,None)).ipython_display()
