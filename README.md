# Human Activity Recognition (HAR) System

## Overview

This project implements a real-time Human Activity Recognition (HAR) system using advanced deep learning techniques. The system employs a Long-term Recurrent Convolutional Networks (LRCN) model to accurately classify various human activities from video inputs captured via a webcam. The user interface is built using Streamlit, enabling a seamless web-based interaction.

## Features

- **Real-Time Processing**: Captures and processes video feed in real-time.
- **Activity Classification**: Identifies activities such as "Apply Lipstick", "Baby Crawling", "Blowing Candles", "Brushing Teeth", "Cutting In Kitchen", "Haircut", "Hammering", "Head Massage", "Horse Riding", "Jumping Jack", and "Shaving Beard".
- **User-Friendly Interface**: Interactive web application for easy usage.
- **Adaptive Learning**: Capable of continuous learning to enhance accuracy.

## Installation

### Prerequisites

- Python 3.9+
- pip (Python package installer)

### System Dependencies

Ensure you have the following system dependencies installed:
- `libgl1-mesa-glx`
- `libglib2.0-0`

On Ubuntu, install these dependencies using:
```sh
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0


### Python Packages
## Clone the repository:

```git clone https://github.com/yourusername/human-activity-recognition.git cd human-activity-recognition```
Install the required Python packages:

sh
Copy code
pip install -r requirements.txt
Usage
Start the Streamlit app:

sh
Copy code
streamlit run HAR.py
Open your web browser and go to http://localhost:8501 to access the HAR system.

Grant necessary permissions for webcam access.

The video feed will be displayed along with real-time activity classification.

Project Structure
HAR.py: Main application file for running the Streamlit app.
model84.keras: Pre-trained LRCN model for activity recognition.
requirements.txt: List of Python dependencies required for the project.
tests/: Directory containing unit tests for the project.
Development
Running Tests
To run unit tests, execute:

sh
Copy code
pytest tests/
Sample Test Case Table
Test Case ID	Description	Input	Expected Output	Actual Output	Status
TC1	Test video feed capture	Start webcam	Webcam feed is displayed	Webcam feed is displayed	Pass
TC2	Test activity classification "Jumping Jack"	Video of activity	Activity classified as "Jumping Jack"	Activity classified as "Jumping Jack"	Pass
TC3	Test invalid input	Obstructed view	Error message or unknown classification	Error message or unknown classification	Pass
Challenges and Limitations
Predefined Activities: Limited to specific activities trained in the model.
Hardware Dependency: Requires a good quality webcam for optimal performance.
Performance Variability: Real-time performance depends on the hardware capabilities of the system.
Future Work
Expand Activity Set: Incorporate a broader range of human activities.
Model Improvement: Enhance model accuracy with more training data and better algorithms.
Performance Optimization: Improve system performance on lower-end devices.
Multisensory Integration: Integrate data from other sensors for more robust activity recognition.
Contributing
Contributions to enhance the HAR system are welcome. Please fork the repository and submit pull requests for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
Streamlit for the web framework.
TensorFlow/Keras for the deep learning tools.
Dataset providers used for training the LRCN model.
Contact
For questions or support, please contact [your-email@example.com].
