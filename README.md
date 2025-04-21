# Long_Jump_Analysis
Long Jump Performance Analyzer
This Python script analyzes long jump performance using computer vision techniques. It processes video footage of a long jump to extract key performance metrics like take-off angle, jump distance, stride characteristics, and horizontal velocity.

Key Features
Pose Detection: Uses MediaPipe's pose estimation to track the athlete's body positions throughout the jump.

Spatial Calibration: Converts pixel measurements to real-world distances using the athlete's known height.

Phase Detection: Identifies different phases of the jump (approach run, take-off, flight, landing).

Performance Metrics:

Calculates take-off angle from leg extension

Estimates jump distance

Measures stride length and frequency during approach

Computes horizontal velocity before take-off

Visualization: Generates an annotated video with performance metrics and creates performance graphs.

How It Works
Initialization:

Takes a video file and athlete's height as input

Sets up pose detection using MediaPipe

Prepares output video writer

Calibration:

Uses a frame where the athlete is standing upright

Measures pixel distance between head and heel

Calculates meters-per-pixel conversion factor

Video Processing:

Processes each frame to detect pose landmarks

Tracks foot positions to detect ground contact

Identifies take-off and landing points

Calculates take-off angle from leg geometry

Analysis:

Analyzes stride patterns during approach

Calculates jump distance from time in air

Generates performance graphs

Output:

Produces an annotated video with analysis overlay

Creates a summary frame with key metrics

Outputs performance graphs

Technical Components
Computer Vision: OpenCV for video processing

Pose Estimation: MediaPipe for body landmark detection

Data Analysis: NumPy for calculations

Visualization: Matplotlib for performance graphs

Command Line Interface: argparse for user input
Output
The script generates:

An annotated video showing pose detection and key events

Performance metrics overlay

Stride analysis graphs

A summary frame with all calculated metric
