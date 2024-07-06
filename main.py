import streamlit as st
import os
os.environ["PAFY_BACKEND"] = "internal"
import cv2
import random
import pafy
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from pytube import YouTube
from keras.models import load_model
from collections import deque
import pickle
from pathlib import Path

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

CLASSES_LIST = [
    "ApplyLipstick",
    "BabyCrawling",
    "BlowingCandles",
    "BrushingTeeth",
    "CuttingInKitchen",
    "Haircut",
    "Hammering",
    "HeadMassage",
    "HorseRiding",
    "JumpingJack",
    "ShavingBeard"
]

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter Object to store the output video in the disk.
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:

            LRCN_model = load_model("model84.keras")

            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(cv2.resize(frame, (original_video_width, original_video_height)))

    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()


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

def process_input(videolink):
    # Process the input text here
    st.write("Input Text:", videolink)


st.title("Human Activity Recognition")

videolink=st.text_input("Enter the youtube video link")

if st.button("Recognize"):
    process_input(videolink)
    current_directory = os.getcwd()
    # Make the Output directory if it does not exist
    test_videos_directory = current_directory
    os.makedirs(test_videos_directory, exist_ok = True)

    # Download a YouTube Video.
    video_title = download_youtube_videos(videolink, test_videos_directory)

    # Get the YouTube Video's path we just downloaded.
    input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

    # Construct the output video path.
    output_video_file_path = f'{test_videos_directory}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'

    # Perform Action Recognition on the Test Video.
    predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

    # Write the video file with the specified frame rate directly
    output_clip = VideoFileClip(output_video_file_path, audio=False, target_resolution=(300, None))

    # Write the video file with the specified frame rate
    output_clip.write_videofile("output_video.mp4")

    # Display the output video.
    st.video("output_video.mp4")


st.write("Sometimes the website may crash")
st.write("Demo video of how the project work")
st.video("Media1.mp4")
