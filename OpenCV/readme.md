# Lane Detection Using OpenCV and MoviePy

This project demonstrates lane detection on road videos using OpenCV for image processing and MoviePy for video processing. The goal is to identify road lanes in video frames and overlay them with graphical markers for visualization.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Required Libraries](#required-libraries)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Code Explanation](#code-explanation)
  - [Key Functions](#key-functions)

---

## Introduction

Lane detection is a critical feature in autonomous vehicles and advanced driver assistance systems (ADAS). This project uses image processing techniques such as edge detection, region masking, and Hough transforms to detect and highlight lanes in a video.

---

## Features

- Detect lanes in video frames using OpenCV.
- Overlay detected lanes on the original video.
- Save the processed video with lane markings.

---

## Installation

### Prerequisites

- Python 3.8 or above
- pip (Python package manager)

### Required Libraries

Install the following libraries before running the project:

```bash
pip install numpy opencv-python moviepy
```
---

## Usage

1. Place the input video in the `data/test_videos` folder.
2. Update the input video path in the script (`VideoFileClip` section).
3. Run the script using the following command:

```bash
python lane_detection.py
```

4. The processed video will be saved in the `test_videos_output` folder.

---

## File Structure

The project directory is organized as follows:

```plaintext
.
├── data/
│   ├── test_videos/
│   │   └── solidYellowLeft.mp4  # Input video file
│   └── test_videos_output/
│       └── solidYellowLeft.mp4  # Output video file
├── lane_detection.py            # Main script for lane detection
└── README.md                    # Project documentation
```

---

## Code Explanation

### Key Functions

1. **`processImage(image)`**:
   - Processes each video frame:
     - Applies region-of-interest masking.
     - Filters colors to isolate lane lines.
     - Converts the image to grayscale.
     - Applies edge detection using the Canny algorithm.
     - Uses the Hough transform to detect lane lines.
     - Overlays the detected lanes on the original frame.

2. **`VideoFileClip`**:
   - Reads the input video frame by frame.

3. **`clip.fl_image(processImage)`**:
   - Applies the `processImage` function to each video frame.

4. **`clip.write_videofile(output1, audio=False)`**:
   - Saves the processed video to the specified output folder.

