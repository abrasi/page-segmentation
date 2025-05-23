# A4 Document Segmentation and Cleaning

This repository provides a Python script using OpenCV to segment, clean, and extract text from A4 paper documents, even when rotated or placed on irregular surfaces.

### Features
- A4 paper contour detection  
- Perspective and rotation correction  
- Background noise and stain removal  
- Text extraction and placement over a clean white background

### Requirements
- Python 3.x  
- OpenCV (`opencv-python`)

### Usage
Images should be placed in the `materiales/` directory.  
Processed results will be saved to `materiales/rebuilt/`.

```bash
python processing.py
