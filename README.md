# Image-Captioning-and-Segmentation
This project combines Image Captioning and Instance Segmentation to automatically generate captions for images and highlight objects within them. It supports both static image upload and real-time webcam input.

🚀 Features

Image Captioning: Generates descriptive captions using a trained deep learning model (InceptionV3 + LSTM).
Instance Segmentation: Uses Detectron2 with pre-trained COCO weights for object detection and segmentation.
Dual Input Modes:
Upload an image for captioning and segmentation.
Use live webcam feed for real-time caption + segmentation overlay.
Overlay Support: Captions and segmentation masks are displayed directly on images/video frames.

🛠️ Tech Stack

Python 3
TensorFlow / Keras – Image feature extraction & caption generation
Detectron2 – Instance segmentation
OpenCV – Webcam integration & display
MS COCO Dataset – Used for training and segmentation backbone
PIL / NumPy – Image processing utilities

📂 Project Structure

├── main.py                 # Main script (image & webcam modes)
├── caption_decoder_weights.h5   # Trained captioning model weights (not included in repo)
├── tokenizer.pickle             # Trained tokenizer (not included in repo)

⚙️ Setup Instructions

1. Clone the Repository

git clone https://github.com/your-username/image-captioning-segmentation.git
cd image-captioning-segmentation

2. Install Dependencies

pip install -r requirements.txt
(Ensure you have Detectron2 installed. For installation instructions, see the Detectron2 docs in google).

3. Add Model Weights & Tokenizer

Place your trained caption_decoder_weights.h5 in the project root.
Place your trained tokenizer.pickle in the project root.

4. Run the Project

For image mode:
python main.py --mode image --image_path sample.jpg

For webcam mode:
python main.py --mode webcam --device 0
(Press q to quit webcam mode)

📊 Training Details

Dataset: MS COCO dataset
Feature Extractor: InceptionV3 (pre-trained on ImageNet)
Captioning Model: LSTM decoder with embedding + dense layers
Segmentation: Mask R-CNN (R-50-FPN) from Detectron2 with COCO weights

🎯 Results

Generates meaningful captions for uploaded images.
Overlays segmentation masks and captions on both static images and live webcam feeds.

MS COCO Dataset - https://cocodataset.org/
Detectron2 - https://github.com/facebookresearch/detectron2
TensorFlow / Keras - https://www.tensorflow.org/
