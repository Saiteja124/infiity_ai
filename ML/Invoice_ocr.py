from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import os
import re
import csv
import spacy
import numpy as np
from pdf2image import convert_from_bytes
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient
import io
from PIL import Image

# Set your Azure service key and endpoint
azure_endpoint = 'https://sabio-khub.cognitiveservices.azure.com/'
azure_key = '519cb0e5e64f4cc591008c74e6bfe5df'

# Create a FormRecognizerClient
form_recognizer_credential = AzureKeyCredential(azure_key)
form_recognizer_client = FormRecognizerClient(azure_endpoint, form_recognizer_credential)

app = Flask(__name__)


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
# extracted_text=""
# df = pd.read_csv(r'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\Source\project-7-at-2023-10-25-15-38-70d0741e.csv')
# df_ = df.drop(labels=[1, 3], axis=0).reset_index(drop=True)
# # print(df_)
# text1 = df_.text.to_list()
# df_pred = df.loc[[1, 3]].reset_index(drop=True)
# text = df_pred.text.to_list()
# for tex in text:
#     extracted_text += extracted_text+tex

# Extracts entities from the input text using a SpaCy model and updates a DataFrame with the results.
def process_text(text, result_df):

    # Load the SpaCy model
    nlp = spacy.load('model/model-best')

    # Extract entities and labels
    doc = nlp(text)
    # print(doc)
    # Extract entities
    # for ent in doc.ents:
    #     print(f"Entity: {ent.text}, Label: {ent.label_}")

    # Get the column names from the existing result_df
    columns = list(result_df.columns)

    # Initialize entity_values with None values for each column
    entity_values = {label: [] for label in columns}

    # Process each entity in the SpaCy doc
    for ent in doc.ents:
        label = ent.label_
        value = ent.text.strip()  # Strip leading/trailing whitespaces

        # Apply specific validation patterns based on the entity label
        if label == 'Line_Amount':
            if value !='$':
                pattern=r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
            if re.match(pattern,value):
                entity_values[label].append(value)
        elif label=='Price':
            if value !='$':
                pattern= r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$'
            if re.match(pattern,value):
                entity_values[label].append(value)

        elif label =='UPC':
            pattern=r'\d{8,}'
            if re.match(r'\d{8,}', value):
                entity_values[label].append(value)

        elif label in columns:
            # If the entity label is a column in the DataFrame, append the value
            entity_values[label].append(value)
        else:
            # If the key doesn't exist, create a new list with the value
            entity_values[label] = [value]

    # Convert the entity_values dictionary to a DataFrame row
    row_df = pd.DataFrame.from_dict(entity_values,orient='index')
    # print(row_df)

    # Update the result_df with the new row of extracted entities
    result_df = row_df.transpose()
    # print(result_df.columns)

    # list of keywords
    keywords = ["Store Address", "ENTERPRISES", "Bill To", "Ship To", "Inc", "Corp", "Ltd", "LLC", "Plc", "L.P.", "LLP", "Ltd.", "LC", "PLLC"]

    # Initialize an empty list to store the extracted text
    extracted_text = []

    # Iterate over all pairs of different keywords (forward direction)
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            # Create a regular expression that captures the text between the current pair of keywords
            keyword1, keyword2 = keywords[i], keywords[j]
            pattern = rf"\b{re.escape(keyword1)}\b(.*?){re.escape(keyword2)}\b"

            # Find all matches of the regular expression in the text with case-insensitivity
            matches = re.findall(pattern, text, re.IGNORECASE)

            # Extract the captured text, including the starting and ending positions, and add it to the extracted_text list
            for match in matches:
                start_position =  text.find(match)

                # Check if the captured text is not empty and meets the length condition before appending
                if match.strip() and len(match) <= 60 and match not in extracted_text:
                    extracted_text.append((start_position, match+keyword2))

    # Iterate over all pairs of different keywords (reverse direction)
    for i in range(len(keywords)-1, 0, -1):
        for j in range(i - 1, -1, -1):
            # Create a regular expression that captures the text between the current pair of keywords
            keyword1, keyword2 = keywords[i], keywords[j]
            pattern = rf"\b{re.escape(keyword1)}\b(.*?){re.escape(keyword2)}\b"

            # Find all matches of the regular expression in the text with case-insensitivity
            matches = re.findall(pattern, text, re.IGNORECASE)

            # Extract the captured text, including the starting and ending positions, and add it to the extracted_text list
            for match in matches:
                start_position =  text.find(match)
                
                # Check if the captured text is not empty and meets the length condition before appending
                if match.strip() and len(match) <= 60 and match not in extracted_text:
                    extracted_text.append((start_position, match+keyword2))

    print("extracted_text",extracted_text)

    # This is the sublist to compare
    sublist = ["Store Address","BILL TO", "SHIP TO"]

    # Create a regular expression pattern for any of the sublist terms
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, sublist)) + r')\b', re.IGNORECASE)

    # Find all occurrences of the pattern in the text
    sublist_positions = [(text.find(match),match.strip()) for match in pattern.findall(text)]

    # print("sublist_positions",sublist_positions)

    # Filter sublist_positions based on the condition related to extracted_text
    filtered_text = []

    for start_position, txt in extracted_text:
        if any(start_position >= position[0] for position in sublist_positions):
            if txt.strip() not in filtered_text:
                filtered_text.append(txt.strip())

    # Print the filtered sublist positions
    print("filtered text ",filtered_text)

    # fill 'Buyer_Name' column with the extracted names
    if result_df['Buyer_Name'].isnull().all() and filtered_text:
        result_df['Buyer_Name'] = filtered_text[0]  # Assuming you want to use the first extracted name


    # Initialize invoice_date to None
    invoice_date = None

    # Extract Invoice Date
    # order_date_match = re.search('(?:Order|Invoice) Date:*-*\s*(\d{1,2}/\d{1,2}/\d{4})|\b(\d{1,2}/\d{1,2}/\d{4})\b', text, re.IGNORECASE)
    # order_date_match = re.search('(?:Order|Invoice)* Date:*-*\s*(\d{1,2}[./]\d{1,2}[./]\d{4})|(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},?\s*\d{4}\b))', text, re.IGNORECASE)

    order_date_match = re.search('(?:Order|Invoice) Date:*-*\s*((?:\d{1,2}[./]\d{1,2}[./]\d{4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2},?\s*\d{4})', text, re.IGNORECASE)

    if order_date_match:
        invoice_date = order_date_match.group(1) 

    print('Invoice Date : ', invoice_date)
    
    # Fill 'Invoice_Date' column with the extracted invoice date
    if result_df['Invoice_Date'].isnull().all() and invoice_date:
        result_df['Invoice_Date'] = invoice_date

    # Initialize email_address to None
    seller_email = None

    # Extract Email Address
    # email_match = re.search(r'\b[A-Za-z\.*A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    email_match = re.search(r'\b[A-Za-z\.*A-Za-z0-9._%+-]+@[A-Za-z.-]+(?:\s?[A-Z|a-z]{2,}\.)?[A-Z|a-z]{2,}\b',text, re.IGNORECASE)

    if email_match:          
        seller_email = email_match.group()

    print('Extracted Email Address: ', seller_email)

    # Fill 'Email_Address' column with the extracted email address
    if result_df['Seller_Email'].isnull().all() and seller_email:
        result_df['Seller_Email'] = seller_email

    # Initialize phone_number to None
    seller_phone = None

    # Extract Phone Number
    phone_match = re.search(r'\b(?:tel|ph(?:one)?)\s*:\s*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b', text, re.IGNORECASE)
    if phone_match:
        seller_phone = phone_match.group(1)

    print('Extracted Phone Number: ', seller_phone)

    # Fill 'Phone_Number' column with the extracted phone number
    if result_df['Seller_Phone'].isnull().all() and seller_phone:
        result_df['Seller_Phone'] = seller_phone
    
    # Initialize invoice_number to None
    invoice_number = None

    # Generic Invoice Number Regex Pattern
    # invoice_pattern = r'(?i)\binvoice\s*\:*\#*([\w\d\s]*?)\s*\d\b'
    # invoice_pattern = r'(?i)\b(?:invoice\/order\s*no|invoice\s*no|invoice)\s*\:*\#*([\w\d\s]*?)\s*\d\b'
    invoice_pattern = r'(?i)\b(?:invoice\/order\s*no|invoice\s*no|invoice)\s*[:#]*\s*([\w\d\s-]*?)\s*\d\b'


    # invoice_pattern = r'(?i)\b(?:invoice|order\s*no|invoice\s*no|#)\s*[:]?[^\w\d\s]*([\w\d]+)\b'
    
    # Extract Invoice Number
    invoice_match = re.search(invoice_pattern, text)
    if invoice_match:
        invoice_number = invoice_match.group(1)

    print('Extracted Invoice Number:', invoice_number)

    # Fill 'Invoice_Number' column with the extracted invoice number
    if result_df['Invoice_Number'].isnull().all() and invoice_number:
        result_df['Invoice_Number'] = invoice_number

    # print(result_df)

    # print(result_df['Product_Code'])
    # Extract UPC and Product_Code from 'Product_Code' column in result_df
    prod = []  # List to store non-numeric Product_Codes
    upc = []   # List to store valid UPCs
    for value in (list(result_df['Product_Code'])):
        if value is None:
            continue
        value = str(value)
        if value.isdigit() and len(value) >= 8 :
            upc.append(value)
           
        else:
            prod.append(value)

    # Update 'UPC' and 'Product_Code' columns in result_df
    result_df['UPC'] = pd.Series(upc)
    result_df['Product_Code'] = pd.Series(prod)

    # Extract context around Line_Amount occurrences in the text
    sub_string_dict={} # Dictionary to store line_amount occurrences and their context
    for line_amount in list(result_df.Line_Amount):
        # Skip if line_amount is None
        if line_amount is None:
            continue
        # Find all occurrences of line_amount in the text and store their indices
        sub_indices = re.finditer(line_amount, text)
        # print(sub_indices)
        for match in sub_indices:
            # print(match)
            index = match.start()
            # sub_string = text[max(index - 80, 0):index].strip()
            # print(sub_string)
            # Store the context (substring) around the line_amount occurrence
            sub_string_dict[index] = line_amount

    # print(result_df.Line_Amount,len(result_df.Line_Amount))
    # print('sub_string_dict',sub_string_dict,len(sub_string_dict))

    # Sort the indices of line_amount occurrences
    sorted_keys = sorted(sub_string_dict.keys())
    # print('sorted_keys',sorted_keys,len(sorted_keys))

    # Extract text around each sorted key and store it in descri_list
    descri_list = []

    if not sorted_keys:
        print("")
    else:
        for key in sorted_keys:
            if key is None:
                continue
            # Extract a substring of fixed length (70 characters) centered around the line_amount occurrence
            extracted_text = text[key - 70:key].strip()
            descri_list.append(extracted_text)
        if not descri_list:
            print('--')

    # print(result_df)
    # print('extracted_text',descri_list,len(descri_list))
    # print(descri_list)


    # Load keywords from CSV
    with open('Source/Unwanted_words.csv', 'r') as csvfile:
        keyword_reader = csv.reader(csvfile)
        keywords = [row[0] for row in keyword_reader]
 
    # Iterate through descri_list and replace unwanted words with spaces
    for i in range(len(descri_list)):
        for keyword in keywords:
            # print(keyword,'--')
            descri_list[i] = descri_list[i].replace(keyword,'')
    # Remove extra spaces resulting from elimination
    descri_list = [' '.join(sentence.split()) for sentence in descri_list]

    # Print the modified descri_list
    # print(descri_list)

    # List to store processed descriptions
    final_desc=[]
    spl='[@_-!*$%^&()<>?/\|}{~:]#' # Set of special characters to consider

    # Iterate through each description in descri_list
    for item in descri_list:
        # Skip if the description is NaN
        if item is np.nan:
            continue
        # Remove specific characters and leading/trailing whitespaces
        result = re.sub(r"[\.\|\*\$,]", "", item).strip()
        
        # Split the result into a list of words
        result_list=result.split(' ')
        result_list[0] = re.sub(r"^\d+\s*","",result_list[0])
        if len(result_list[0])<=2:
            result=' '.join(result_list[1:])
        # print(result,'++')
        for i in range(10):
            if result[0].isalpha() == False :
                result = re.sub(r"^\d+\s*", "", result)
            else:
                break
        # Reverse the result and remove leading digits up to 20 iterations
        result = result[::-1]
        # print(result,'\n','*'*20)

        for i in range(20):
            if result[0].isalpha() == False :
                result = re.sub(r"^\d+\s*", "", result)   
                # print(result)  
            else:
                result = result[::-1]
                break
        # print(result,'__________')
        for i in range(20):
            if result[0] in spl: 
                result = result[::-1]
            else:
                break

        # Append the processed description to the final_desc list
        final_desc.append(result)
    # sys.exit('done')
    # print('final_desc--------',final_desc)
    # print(final_desc)
    # Drop the existing 'Description' column from result_df
    result_df.drop('Description',axis=1,inplace=True)
    # Create a new DataFrame with the processed descriptions and concatenate it with result_df
    df_des1=pd.DataFrame({'Description':final_desc})
    result_df=pd.concat([result_df,df_des1], axis=1)
    
    return result_df

# Your list of text data
# df = pd.read_csv(r'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\Source\project-7-at-2023-10-25-15-38-70d0741e.csv')
# df_ = df.drop(labels=[1, 3], axis=0).reset_index(drop=True)
# print(df_)
# text1 = df_.text.to_list()
# df_pred = df.loc[[1, 3]].reset_index(drop=True)
# text = df_pred.text.to_list()

# # Process each text file and save the result to a CSV file
# for idx, text_data in enumerate(text):
#     # Create an empty DataFrame for each iteration
#     result_df = pd.DataFrame(columns=['Product_Code', 'Description', 'UPC', 'Quantity', 'Price',
#                                    'Line_Amount', 'Invoice_Amount','Subtotal','Total_Tax','Seller_Name', 'Seller_Address',
#                                    'Buyer_Name', 'Buyer_Address', 'Invoice_Number', 'Invoice_Date',
#                                    'Seller_Phone', 'Seller_Email','Po_Number','Seller_Website','Seller_Fax_Number','Shipto_Name',
#                                    'Shipto_Address'])

#     result_df = process_text(text_data, result_df)

#     # Specify the output file path
#     # output_path = rf'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Khub\predictions1\pred_trained_{idx + 1}.csv'


#     # Check if the file already exists, and if it does, remove it
#     if os.path.exists(output_path):
#         os.remove(output_path)

    # Save the DataFrame to a CSV file with the specified path
    # result_df.to_csv(output_path, index=False)

    # print(f"CSV file saved to: {output_path}")

def specific_fields(texts, fields):
    # Initialize an empty DataFrame to store the extracted data
    result_df = pd.DataFrame(columns=['Product_Code', 'Description', 'UPC', 'Quantity', 'Price',
                                       'Line_Amount', 'Invoice_Amount', 'Subtotal', 'Total_Tax', 'Seller_Name',
                                       'Seller_Address', 'Buyer_Name', 'Buyer_Address', 'Invoice_Number',
                                       'Invoice_Date', 'Seller_Phone', 'Seller_Email', 'Po_Number',
                                       'Seller_Website', 'Seller_Fax_Number', 'Shipto_Name', 'Shipto_Address'])

    # Define a DataFrame to store the extracted columns
    extracted_columns_df = pd.DataFrame()
    # Iterate over each text in the list
    for idx, text_data in enumerate(texts):
        # Process the text and update the result DataFrame
        result_df = process_text(text_data, result_df)

        # # Define a DataFrame to store the extracted columns
        # extracted_columns_df = pd.DataFrame()

        # Iterate over the requested fields
        for field in fields:
            # Check if the field exists in the DataFrame
            if field in result_df.columns:
                # Add the specified column to the DataFrame
                extracted_columns_df[field] = result_df[field]
        
        return extracted_columns_df

        # Specify the output file path
        # output_path = output_path = rf'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\ML\extracted_specific_columns{idx+1}.csv'

        # Check if the file already exists, and if it does, remove it
        # if os.path.exists(output_path):
        #     os.remove(output_path)

        # Save the DataFrame to a CSV file
        # extracted_columns_df.to_csv(output_path, index=False)

# fields = ['Description', 'Product_Code', 'Line_Amount']  # User-defined fields
# output_file = rf'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Integrating_webpage\ML\extracted_columns_.csv'
# process_text_and_save_csv(text, fields)