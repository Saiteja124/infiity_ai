from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for,abort
import pandas as pd
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.formrecognizer import FormRecognizerClient
import io
from PIL import Image
import os
import cv2
# Set your Azure service key and endpoint
# azure_endpoint = 'https://sabio-khub.cognitiveservices.azure.com/'
# azure_key = '519cb0e5e64f4cc591008c74e6bfe5df'

# # Create a FormRecognizerClient
# form_recognizer_credential = AzureKeyCredential(azure_key)
# form_recognizer_client = FormRecognizerClient(azure_endpoint, form_recognizer_credential)

import sys
sys.path.append('ML/Invoice_ocr.py')  # Adjust the path accordingly
sys.path.append('database')
from ML.Invoice_ocr import process_text,specific_fields
from database.create_db import register_user,login_user,user_exists,email_exists

app = Flask(__name__,template_folder='templates')

# def process_pdf(uploaded_file):
    # Convert PDF to images
    # images = convert_from_bytes(uploaded_file.read(), dpi=200, poppler_path=r'C:\Program Files\poppler-23.08.0\Library\bin')

    # # Extract text from each image using Azure Form Recognizer
    # extracted_text = ""
    # for idx, image in enumerate(images):
    #     # Convert the image to grayscale
    #     grayscale_image = image.convert('L')

    #     # Resize the grayscale image to fit within 4MB limit
    #     max_image_size = (2048, 2048)  # Set the maximum image size as needed
    #     grayscale_image.thumbnail(max_image_size, Image.LANCZOS)

    #     # Convert the resized grayscale image to bytes
    #     image_bytes = io.BytesIO()
    #     grayscale_image.save(image_bytes, format='PNG')
    #     image_bytes.seek(0)

    #     # OCR with Azure Form Recognizer
    #     poller = form_recognizer_client.begin_recognize_content(form=image_bytes, content_type="image/png")
    #     form_pages = poller.result()

    #     poller.wait()

        # Extracted text from Form Recognizer response
        # for page in form_pages:
        #     extracted_text += " ".join([line.text for line in page.lines])
extracted_text=""
df = pd.read_csv('Source/project-7-at-2023-10-25-15-38-70d0741e.csv')
df_ = df.drop(labels=[1, 3], axis=0).reset_index(drop=True)
# print(df_)
text1 = df_.text.to_list()
# print(text1)
df_pred = df.loc[[1, 3]].reset_index(drop=True)
text = df_pred.text.to_list()
for tex in text:
    extracted_text += extracted_text+tex
# print(extracted_text)

# names = ["dog","person","cat","tv","car","meatballs","marinara sauce","tomato soup","chicken noodle soup",
# "french onion soup","chicken breast","ribs","pulled pork","hamburger","cavity","Ac","aeroplane","lion","elephant",
# "zebra","monkey","tiger","jiraffe","fox","apple","knife","bag","school bag","table","ball","banana","bat","bed",
# "potted plant","frame","lamp","window","beetroot","bench","trees","bicycle","helmet","boat","ship","book",
# "bottle gourd","bottle","hat","watch","mobile","saftey vest","bus","cabbage","capsicum","carrot","cauliflower",
# "chair","laptop","clock","cock","cooler","deer","doll","dragon fruit","duck","fan","lights","flute","glass",
# "grocery store","guitar","gun","helicopter","hen","horse","house","bike","jar","kite","light","sofa","mango",
# "microwave oven","tomato","mushrooms","onion","orange","parachute","peacock","pen","pencil","sharpener","eraser",
# "pine apple","pomegranate","potato","refrigerator","ridge gourd","skateboard","stove","strawberry","cup",
# "tooth bresh","traffic lights","truck","umbrella","utensils","water tanker","watermelon"
# ]

# print(len(names))
# registered_users = []

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def registration():
    if request.method == 'POST':
        username = request.json.get('username')
        email = request.json.get('email')
        password = request.json.get('password')

        # Check if username and email already exist
        username_exist = user_exists(username)
        email_exist = email_exists(email)

        if username_exist and email_exist:
            return jsonify({'registration_success': False, 'username_exists': True, 'email_exists': True})
        elif username_exist:
            return jsonify({'registration_success': False, 'username_exists': True, 'email_exists': False})
        elif email_exist:
            return jsonify({'registration_success': False, 'username_exists': False, 'email_exists': True})
        else:
            # Call the register_user function from create_db.py
            registration_success = register_user(username, email, password)

            if registration_success:
                return jsonify({'registration_success': True})
            else:
                return jsonify({'registration_success': False})

    # Render the registration form for GET requests
    return render_template('registration.html')



@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = request.json.get('username')  # Update to use JSON data
        password = request.json.get('password')  # Update to use JSON data

        # Use the login_user function from create_db.py
        if login_user(username, password):
            # Login successful, respond with success status
            return jsonify({'success': True})
        else:
            # Login failed, respond with failure status
            return jsonify({'success': False})

    # Render the login form for GET requests
    return render_template('login.html')


@app.route('/Sabio_use_cases')
def upload_file():
    return render_template('usecases.html')

# @app.route('/upload_pdf')
# def upload_pdf():
#     return render_template('upload_pdf.html')
@app.route('/upload_pdf')
def upload_pdf():
    return render_template('ocr application.html')

# Flag to track if PDF has been processed
# pdf_processed = False
@app.route('/process_text', methods=['POST'])
def process_text_route():
    global pdf_processed
    if request.method == 'POST':
        try:
            # Get the input text from the form
            # input_text = request.form['input_text']
            # Get the uploaded PDF file from the request
            # pdf_file = request.files['pdf_file']

            # Create an empty DataFrame for the result
            result_df = pd.DataFrame(columns=['Product_Code', 'Description', 'UPC', 'Quantity', 'Price',
                                              'Line_Amount', 'Invoice_Amount', 'Subtotal', 'Total_Tax',
                                              'Seller_Name', 'Seller_Address', 'Buyer_Name', 'Buyer_Address',
                                              'Invoice_Number', 'Invoice_Date', 'Seller_Phone', 'Seller_Email',
                                              'Po_Number', 'Seller_Website', 'Seller_Fax_Number', 'Shipto_Name',
                                              'Shipto_Address'])

            # Process the input text using your function
            # result_df = process_text(input_text, result_df)
            result_df=process_text(extracted_text, result_df)
            pdf_processed = True

            # Save the result to a CSV file
            result_csv_path = 'output/processed_result.csv'
            
            result_df.to_csv(result_csv_path, index=False)

            # Assuming 'success' is True if processing is successful
            return jsonify({'success': True})

            

            # Save the result to a CSV file
            # result_csv_path = 'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\output\processed_result.csv'
            
            # result_df.to_csv(result_csv_path, index=False)

            # Save the result to a CSV string
            # result_csv_string = result_df.to_csv(index=False)
            
            # Send the CSV file as an attachment in the response
            # return send_file(result_csv_path, as_attachment=True, download_name='processed_result.csv')

        except Exception as e:
            # Log the error for your reference
            app.logger.error(f"Error processing text: {str(e)}")
            return jsonify({'success': False, 'error': 'An error occurred while processing the text.'})

# @app.route('/get_column_names')
# def get_column_names():
#     # Assuming you have a function to retrieve column names from your data
#     column_names = ['Product_Code', 'Description', 'UPC', 'Quantity', 'Price',
#                                        'Line_Amount', 'Invoice_Amount', 'Subtotal', 'Total_Tax', 'Seller_Name',
#                                        'Seller_Address', 'Buyer_Name', 'Buyer_Address', 'Invoice_Number',
#                                        'Invoice_Date', 'Seller_Phone', 'Seller_Email', 'Po_Number',
#                                        'Seller_Website', 'Seller_Fax_Number', 'Shipto_Name', 'Shipto_Address']
#     return jsonify({'columnNames': column_names})       
       
@app.route('/process_specific_fields', methods=['GET', 'POST'])
def process_specific_fields():
    # if request.method == 'GET':
    #     return render_template('specific_fields.html')
    if request.method == 'GET':
        return render_template('Specific fields.html')
    elif request.method == 'POST':
        # field1 = request.form.get('fields')
        # field2 = request.form.get('fields')
        # fields = [field1, field2]
        fields_input = request.form.get('fields')
        fields = [field.strip() for field in fields_input.split(',')]  # Splitting fields by comma

        # Call specific_fields function to get the DataFrame
        extracted_df = specific_fields(text, fields)


        # Write the DataFrame to a CSV file
        output_file = 'output/extracted_columns.csv'
            

        # Write the DataFrame to a CSV file
        # output_file = 'extracted_columns.csv'
        extracted_df.to_csv(output_file, index=False)
        return send_file(output_file, as_attachment=True)
# @app.route('/process_specific_fields')
# def process():
#     # return render_template('specific_fields.html')
#     field1 = request.form.get('fields')
#     field2 = request.form.get('fields')
#     fields = [field1,field2]

#     # # Call specific_fields function to get the DataFrame
#     extracted_df = specific_fields(text, fields)
    
#     # # Write the DataFrame to a CSV file
#     output_file = 'extracted_columns.csv'
#     extracted_df.to_csv(output_file, index=False)
#     return send_file(output_file, as_attachment=True)
    # return render_template('specific_fields.html')

# @app.route('/download_csv')
# def download_csv():
#     result_csv_path = 'output/processed_result.csv'
#     return send_file(result_csv_path, as_attachment=True)

# pdf_processed = False
@app.route('/download_csv', methods=['GET'])
def download_csv():
    global pdf_processed
    if pdf_processed:
        try:
            result_csv_path = 'output/processed_result.csv'
            return send_file(result_csv_path, as_attachment=True, download_name='processed_result.csv')
        except Exception as e:
            app.logger.error(f"Error downloading CSV: {str(e)}")
            return jsonify({'success': False, 'error': 'An error occurred while downloading the CSV file.'})
        finally:
            pdf_processed = False  # Reset the flag after downloading
    else:
        return jsonify({'success': False, 'error': 'PDF has not been processed yet.'})
          
# @app.route('/download_csv')
# def download_csv():
#     global pdf_processed
#     result_csv_path = 'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\output\processed_result.csv'
    
#     # Check if the PDF has been processed before allowing download
#     if pdf_processed and os.path.exists(result_csv_path):

#         # Reset the flag after the download
#         pdf_processed = False
#         # If the PDF has been processed and the CSV file exists, send it for download
#         return send_file(result_csv_path, as_attachment=True, download_name='processed_result.csv')
#     else:
#         # If the PDF has not been processed or the CSV file doesn't exist, return an error or redirect as needed
#         abort(404)  # For example, return a 404 error indicating the file is not found

# @app.route('/check_pdf_processed_status', methods=['GET'])
# def check_pdf_processed_status():
#     global pdf_processed
#     # Return the PDF processing status as JSON
#     return jsonify({'pdf_processed': pdf_processed})

# @app.route('/object_recognization', methods=['GET','POST'])
# def obj_recognition():
#     return render_template('obj recognization.html')

# from ultralytics import YOLO
# model = YOLO("C:\PRASANNA_SABIO_FILES\Object detection\yolov8m_custom.pt")
# model = YOLO(r"C:\Users\posha\Downloads\best (3).pt") 
# Load YOLOv8 model
# model = YOLO("yolov8n.pt", "v8")

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/detect', methods=['GET','POST'])
# def detect_objects():
#     # file = request.files['file']
#     if request.method=='POST':
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
#         image = cv2.imread(file)

#         # Perform object detection
#         detection_output = model.predict(source=image, save=True, show=True)
#         cv2.waitKey(0)

#         # Convert detection_output to a dictionary
#         # detection_dict = {'predictions': detection_output.predictions.tolist()}

#         # Return detection_dict as JSON response
#         # return jsonify(detection_dict)

#         # return detection_output
#         return jsonify(detection_output)
#     return render_template('obj recognization.html')



from ultralytics import YOLO
from ML.custom_model import video_detection
from flask_wtf import FlaskForm
from flask import session, Response, send_file


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange

app.config['SECRET_KEY'] = 'sabioinfotech'
app.config['UPLOAD_FOLDER'] = 'static/files'

import tempfile
import numpy as np

class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])        
@app.route('/upload_pdf', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('upload_pdf.html')
        
@app.route('/detect', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('obj recognization.html', form=form)

@app.route('/video')
def video():

    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')


import google.generativeai as palm

api_key = 'AIzaSyCiVyNk7AdSOdrPFE8eOV2-5xT0lmHgzco'
palm.configure(api_key=api_key)

@app.route('/question')
def index():
    return render_template('que_llm.html')

@app.route('/generate_response', methods=['GET','POST'])
def generate_response():
    if request.method == 'POST':
        question = request.form['question']
        
        # Generate response using the Google Generative AI
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt=question,
            temperature=0.5
        )
        
        response = completion.result
        
        response = response.replace('*', '')
        
        # Truncate response to 200 words
        response_words = response.split()[:200]
        truncated_response = ' '.join(response_words)

        # Return a JSON response
        return jsonify(question=question, truncated_response=truncated_response)
        
    return render_template('llm.html')
    

# @app.route('/detect', methods=['GET', 'POST'])
# def detect_objects():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
        
#         # Close all OpenCV windows
#         # cv2.destroyAllWindows()

#         # Check if the uploaded file is a video
#         _, file_ext = os.path.splitext(file.filename)
#         if file_ext.lower() in ['.mp4', '.avi', '.mov']:
#             # Save the video file to a temporary location
#             video_path = os.path.join(tempfile.gettempdir(), file.filename)
#             file.save(video_path)

#             # Perform object detection on the video
#             detection_output = model.predict(source=video_path, save=True, show=True)
#             cv2.waitKey(0)

#             # Return the detection output
#             # return jsonify(detection_output)
#         elif file_ext.lower() in ['.png', '.jpg', '.jpeg']:

#             # Save the image file to a temporary location
#             image_path = os.path.join(tempfile.gettempdir(), file.filename)
#             file.save(image_path)

#              # Perform object detection on the image
#             # image = cv2.imread(image_path)
            
#              # Read the image from file
#             image = cv2.imread(image_path)

#             # Resize the image to a specific size (e.g., 800x600)
#             resized_image = cv2.resize(image, (800, 600))

#             # Read image from file
#             # img_bytes = file.read()
#             # nparr = np.frombuffer(img_bytes, np.uint8)
#             # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#             # Perform object detection on the image
#             detection_output = model.predict(source=image_path, save=True, show=True)
#             cv2.waitKey(0)

#             # Convert the Results object to a JSON-serializable format
#             # detection_json = detection_output.dict()
#             # print(detection_json)

#             # Return the detection output
#             # return jsonify(detection_output)
#         else:
#             return jsonify({'error': 'Unsupported file type'})
        
#         # # Convert detection_output to a JSON-serializable format
#         detection_json = [result.dict() for result in detection_output]

#         # # Return the detection output as JSON
#         return jsonify(detection_json)
    
#     return render_template('obj recognization.html')








# @app.route('/object_recognization', methods=['GET','POST'])
# def predict_img():
#     if request.method=='POST':
#         f=request.files["file"]
#         basepath = os.path.dirname(__file__)
#         filepath = os.path.join(basepath,'uploads',f.filename)
#         print("upload folder is ", filepath)
#         f.save(filepath)
#         global imgpath
#         predict_img.imgpath = f.filename
#         print("printing predict_img :::", predict_img)

#         file_extension = f.filename.rsplit('.',1)[1].lower()

#         if file_extension == 'jpg':
#             img =cv2.imread(filepath)
#             frame =cv2.imencode('.jpg',cv2.UMat(img))[1].tobytes()

#             image = Image.open(io.BytesIO(frame))

#             # perform te detection
#             yolo = YOLO('yolov8n.pt')
#             detections=yolo.predict(image,save=True)
#             # return display(f.filename)

#     return render_template('obj recognization.html')







# @app.route('/object_recognization', methods=['GET','POST'])
# def obj_recognition():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if file:
#         # Save the uploaded file temporarily
#         file_path = 'temp_upload.' + file.filename.rsplit('.', 1)[1].lower()
#         file.save(file_path)

#         # Perform object detection
#         detection_output = yolo_model.predict(source=file_path, conf=0.25, save=True)

#         # Return detection results as JSON
#         return jsonify({'detection_output': detection_output})


# if __name__ == '__main__':
#     # Use this for development purposes, switch to a production-ready server for deployment
#     app.run(debug=True)
