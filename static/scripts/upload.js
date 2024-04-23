// document.addEventListener('DOMContentLoaded', function() {
//     const uploadForm = document.getElementById('uploadForm');
//     // const uploadStatus = document.getElementById('uploadStatus');
//     const downloadButton = document.getElementById('downloadCSV');
//     const mainContent = document.querySelector('.main-content');

//     let uploadedFileName = '';

//     uploadForm.addEventListener('submit', function(event) {
//         event.preventDefault();

//         // Check if the user is authenticated before allowing file upload
//         // if (!isAuthenticated) {
//         //     alert('Please log in before uploading a file.');
//         //     return;
//         // }

//         const fileInput = uploadForm.querySelector('#pdfFile');
//         const file = fileInput.files[0];
//         if (file) {
//             // Create a FormData object to send the file to the server
//             const formData = new FormData();
//             formData.append('pdf_file', file);

//             // Use Fetch API to send the file to the server for processing
//             fetch('/process_text', {
//                 method: 'POST',
//                 body: formData
//             })
//             .then(response => response.json())
//             .then(data => {
//                 console.log(data);
//                 // Display the server response or handle further actions
//                 if (data.success) {
//                     // Display processing message
//                     displayMessage('PDF processed successfully.', mainContent);
//                     // message='PDF processed successfully.'
//                     // uploadStatus.innerText = 'PDF processed successfully.';

//                     // Enable the download button
//                     downloadButton.disabled = false;
//                     // Store the uploaded filename for later use
//                     uploadedFileName = file.name;
//                 } else {
//                     // Display error message
//                     displayMessage('Error processing PDF.', mainContent);
//                     // message='Error processing PDF.'
//                     // uploadStatus.innerText = 'Error processing PDF.';
//                     downloadButton.disabled = true;
//                 }
//             })
//             // .catch(error => {
//             //     console.error('Error:', error);
//             //     // message='Error processing PDF.'
//             //     uploadStatus.innerText = 'Error processing PDF file.';
//             // });
//         } else {
//             // message='Please select a PDF file.'
//             displayMessage('Please select a PDF file.', mainContent);
//             // uploadStatus.innerText= 'Please select a PDF file.';
//         }
//     });

//     downloadButton.addEventListener('click', function() {
//         // Make a request to the server to initiate the download
//         fetch('/download_csv', {
//             method: 'GET',
//         })
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error('Download failed. Please try again.');
//             }
//             return response.text(); // Assuming the CSV data is sent as plain text
//         })
//         .then(csvData => {
//             // Create a Blob with the CSV data
//             const blob = new Blob([csvData], { type: 'text/csv' });
            
//             // Create a link element to trigger the download
//             const url = window.URL.createObjectURL(blob);
//             const a = document.createElement('a');

//             // Set the custom filename here (e.g., 'custom_filename.csv')
//             const customFilename = 'C:\PRASANNA_SABIO_FILES\Invoice_Project\Sabio_Intefrating_webpage\output\processed_result.csv';

//             a.href = url;
//             a.download = uploadedFileName.replace('.pdf', '.csv');
//             document.body.appendChild(a);
//             a.click();
//             document.body.removeChild(a);
//             window.URL.revokeObjectURL(url);
//             // Remove the downloading icon after download is complete
//             downloadButton.classList.remove('downloading');
//         })
//         .catch(error => {
//             console.error('Error:', error);
//             alert(error.message);
//         });
//     });
// });


// function displayMessage(message, element) {
//     // Create a div element to display the message
//     const messageDiv = document.createElement('div');
//     messageDiv.textContent = message;

//     // Append the message to the specified element
//     element.appendChild(messageDiv);

//     // Clear the message after a short delay (optional)
//     setTimeout(() => {
//         element.removeChild(messageDiv);
//     }, 3000); // Adjust the delay as needed (e.g., 3000 milliseconds = 3 seconds)
// }


// document.addEventListener('DOMContentLoaded', function() {
//     const uploadForm = document.getElementById('uploadForm');
//     const downloadButton = document.getElementById('downloadCSV');
//     const mainContent = document.querySelector('.upload-pdf');
//     // const mainContent = document.querySelector('.content');
//     const messageContainer = document.querySelector('.message-container');

//     let uploadedFileName = '';

//     uploadForm.addEventListener('submit', function(event) {
//         event.preventDefault();

//         const fileInput = uploadForm.querySelector('#pdfFile');
//         const file = fileInput.files[0];
//         if (file) {
//             const formData = new FormData();
//             formData.append('pdf_file', file);

//             // Display processing message
//             // displayProcessingMessage('Processing PDF...', mainContent);
//             // Display processing message
//             processingMessage = displayProcessingMessage('Processing PDF...', mainContent);

            

//             fetch('/process_text', {
//                 method: 'POST',
//                 body: formData
//             })
//             .then(response => response.json())
//             .then(data => {
//                 console.log(data);
//                 if (data.success) {
//                     displayMessage('PDF processed successfully, download the csv file', messageContainer);
//                     downloadButton.disabled = false;
//                     uploadedFileName = file.name;
//                 } else {
//                     displayMessage('Error processing PDF.', messageContainer);
//                     downloadButton.disabled = true;
//                 }
//             })
//             .catch(error => {
//                 console.error('Error:', error);
//                 displayMessage('Error processing PDF.', messageContainer);
//                 downloadButton.disabled = true;
//             })
//             .finally(() => {
//                 // Remove processing message once processing is finished
//                 // if (processingMessage.parentNode) {
//                 //     processingMessage.parentNode.removeChild(processingMessage);
//                 // }
//                 if (processingMessage && processingMessage.parentNode) {
//                     processingMessage.parentNode.removeChild(processingMessage);
//                 }
//             });
//         } else {
//             displayMessage('Please select a PDF file.', messageContainer);
//         }
//     });

//     downloadButton.addEventListener('click', function() {
//         const downloadingMessage = displayProcessingMessage('Downloading CSV file...', mainContent);

//         fetch('/download_csv', {
//             method: 'GET',
//         })
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error('Download failed. Please try again.');
//             }
//             return response.text();
//         })
//         .then(csvData => {
//             const blob = new Blob([csvData], { type: 'text/csv' });
            
//             const url = window.URL.createObjectURL(blob);
//             const a = document.createElement('a');

//             a.href = url;
//             a.download = uploadedFileName.replace('.pdf', '.csv');
//             document.body.appendChild(a);
//             a.click();
//             document.body.removeChild(a);
//             window.URL.revokeObjectURL(url);
//             downloadButton.classList.remove('downloading');
//             // Remove the downloading message after 3 seconds
//             setTimeout(() => {
//                 if (downloadingMessage.parentNode) {
//                     downloadingMessage.parentNode.removeChild(downloadingMessage);
//                 }
//             }, 3000);
//         })
//         .catch(error => {
//             console.error('Error:', error);
//             alert(error.message);
//         })

//         .finally(() => {
//             // Remove processing message once downloading is complete
//             if (processingMessage && processingMessage.parentNode) {
//                 processingMessage.parentNode.removeChild(processingMessage);
//             }
//         });
//     });
// });

// function displayProcessingMessage(message, element) {
//     const processingMessage = document.createElement('div');
//     processingMessage.textContent = message;
//     processingMessage.classList.add('processing-message');
//     element.appendChild(processingMessage);

//     // Remove the processing message after the specified duration
//     // setTimeout(() => {
//     //     if (processingMessage.parentNode) {
//     //         processingMessage.parentNode.removeChild(processingMessage);
//     //     }
//     // }, 3000);

//     // return processingMessage; // Return the processing message element to be used later
// }

// function displayMessage(message, element) {
//     const messageDiv = document.createElement('div');
//     messageDiv.textContent = message;
//     element.appendChild(messageDiv);
//     // setTimeout(() => {
//     //     element.removeChild(messageDiv);
//     // }, 3000);
// }

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const downloadButton = document.getElementById('downloadCSV');
    const mainContent = document.querySelector('.upload-pdf');
    const messageContainer = document.querySelector('.message-container');

    let uploadedFileName = '';

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const fileInput = uploadForm.querySelector('#pdfFile');
        const file = fileInput.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('pdf_file', file);

            const processingMessage = displayProcessingMessage('Processing PDF...', mainContent);

            fetch('/process_text', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.success) {
                    displayMessage('PDF processed successfully, download the csv file', messageContainer);
                    downloadButton.disabled = false;
                    uploadedFileName = file.name;
                } else {
                    displayMessage('Error processing PDF.', messageContainer);
                    downloadButton.disabled = true;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                displayMessage('Error processing PDF.', messageContainer);
                downloadButton.disabled = true;
            })
            .finally(() => {
                if (processingMessage && processingMessage.parentNode) {
                    processingMessage.parentNode.removeChild(processingMessage);
                }
            });
        } else {
            displayMessage('Please select a PDF file.', messageContainer);
        }
    });

    downloadButton.addEventListener('click', function() {
        const downloadingMessage = displayProcessingMessage('Downloading CSV file...', mainContent);

        // Remove the success message when download button is clicked
        const successMessage = document.querySelector('.success-message');
        if (successMessage && successMessage.parentNode) {
            successMessage.parentNode.removeChild(successMessage);
        }

        fetch('/download_csv', {
            method: 'GET',
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Download failed. Please try again.');
            }
            return response.text();
        })
        .then(csvData => {
            const blob = new Blob([csvData], { type: 'text/csv' });
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');

            a.href = url;
            a.download = uploadedFileName.replace('.pdf', '.csv');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            downloadButton.classList.remove('downloading');
            setTimeout(() => {
                if (downloadingMessage.parentNode) {
                    downloadingMessage.parentNode.removeChild(downloadingMessage);
                }
            }, 3000);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(error.message);
        })

        .finally(() => {
            if (processingMessage && processingMessage.parentNode) {
                processingMessage.parentNode.removeChild(processingMessage);
            }
        });
    });
});

function displayProcessingMessage(message, element) {
    const processingMessage = document.createElement('div');
    processingMessage.textContent = message;
    processingMessage.classList.add('processing-message');
    element.appendChild(processingMessage);
    return processingMessage;
}

function displayMessage(message, element) {
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.classList.add('success-message'); // Add a class for identifying success message
    element.appendChild(messageDiv);
}
