# Sabio Web Application

Welcome to Sabio Web Application, a robust solution for extracting and analyzing data from various types of documents and images. This application includes three main features: Optical Character Recognition (OCR), Specific Field Extraction, and Object Detection.

## System Overview

The web application is built using Python and the Flask web framework. It includes a back-end API for processing requests and a front-end interface for users to upload documents and images, request processing, and view results.

## Features

### 1. Optical Character Recognition (OCR)

The OCR feature extracts text from various types of documents, including PDFs, images, and scanned documents.

#### Architecture Overview – OCR

![image](https://github.com/Sabio-Infotech/Sabio_OCR_Application/assets/142021993/697a68f4-c421-4bdb-96e5-bea633a4642c)

- Preprocessing the document to improve OCR accuracy.
- Using the OCR engine to extract text from the document.
- Post-processing the extracted text to improve readability and format.

### 2. OCR Specific Field Extraction

This feature extracts specific fields from documents. Users can specify the fields they want to extract, and the application performs the extraction process. It involves similar steps as OCR but with a focus on specific fields or columns extraction.

#### Architecture Overview – OCR Specific Field Extraction

![image](https://github.com/Sabio-Infotech/Sabio_OCR_Application/assets/142021993/03591b1e-d49c-42ff-ae0d-5935c0ae0f4d)

- Preprocessing the document to improve the OCR accuracy.
- Using the OCR engine to extract text from the document.
- Post processing the extracted text to extract specific fields or columns.

### 3. Object Detection

Users can upload images containing objects of interest, and the application will detect and highlight those objects within the images. It involves the following steps:

- Preprocessing the image to improve object detection accuracy.
- Using the OpenCV library to detect objects in the image.
- Post-processing the detected objects to improve accuracy and format.

## User Guide

To use the web application:

1. **Sign in to the application:** Users sign in to the application.
2. **Select processing option:** users select the desired processing option (OCR Application, OCR Specific Field Extraction, or Object Detection).
3. **Upload a document or image:** users can upload the corresponding document, image, or video for processing.
4. **Request processing:** After uploading the file, users initiate the processing by submitting the request.
5. **View results:** Once processing is complete, users can view the results, including extracted text, specific fields, or detected objects.

