

# Long Jump Performance Analyzer 🏃‍♂️➡️🏅

![Sample Analysis Output](docs/assets/sample_analysis.gif)

A computer vision tool for analyzing long jump technique and performance metrics using pose estimation.

## Features

- 📏 **Spatial calibration** using athlete's height
- 🔍 **Pose detection** with MediaPipe
- ⏱️ **Phase detection** (approach, take-off, flight, landing)
- 📊 **Performance metrics**:
  - Take-off angle
  - Jump distance
  - Stride length/frequency
  - Horizontal velocity
- 📹 **Annotated video output**
- 📈 **Performance graphs**

## How It Works

```mermaid
graph TD
    A[Input Video] --> B[Pose Detection]
    B --> C[Spatial Calibration]
    C --> D[Phase Detection]
    D --> E[Metric Calculation]
    E --> F[Output Visualization]
