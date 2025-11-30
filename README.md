# Image Captioner with TensorFlow

An end-to-end Deep Learning project that generates descriptive captions for images. This system implements a "Merge" architecture, combining **Computer Vision (ResNet50)** for visual feature extraction and **Natural Language Processing (LSTM)** for text generation.

## Model Architecture

This project uses a hybrid architecture that merges an image encoder with a text decoder:

1.  **Image Encoder (ResNet50):** * Uses **Transfer Learning** with weights pre-trained on ImageNet.
    * The last classification layer is removed to extract a 2048-dimensional feature vector from the bottleneck layer.
    * A Dense layer compresses these features to match the embedding size.
2.  **Text Decoder (LSTM):**
    * Inputs are passed through an **Embedding Layer** using pre-trained **GloVe Vectors (200d)**.
    * Processed by an **LSTM layer** to capture sequence context.
3.  **Merge & Prediction:**
    * The visual and textual features are concatenated.
    * A final Decoder LSTM and Dense layer predict the next word in the sequence (Softmax output over a 5000-word vocabulary).

## Dataset

The model is trained on the **Flickr8k Dataset**, a standard benchmark for image captioning.
* **Images:** 8,000 images depicting various scenes and actions.
* **Captions:** 5 captions per image (40,000 total captions).

