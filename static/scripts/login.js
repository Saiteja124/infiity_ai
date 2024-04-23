
// uploadForm.addEventListener('submit', function(event) {
//     event.preventDefault();
//     const fileInput = uploadForm.querySelector('#pdfFile');
//     const file = fileInput.files[0];
//     if (file) {
//         // You can perform PDF extraction here if necessary
//         uploadStatus.innerText = 'PDF uploaded successfully: ' + file.name;
//         loginForm.querySelector('button[type="submit"]').disabled = true; // Disable the login button after upload
//     } else {
//         uploadStatus.innerText = 'Please select a PDF file.';
//     }
// });


// downloadButton.addEventListener('click', function() {
//     // Generate and download the CSV file
//     // Replace this with your CSV generation logic
//     const csvData = 'Your CSV Data Here';
//     const blob = new Blob([csvData], { type: 'text/csv' });
//     const url = window.URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = uploadedFileName.replace('.pdf', '.csv');
//     a.download = 'data.csv';
//     document.body.appendChild(a);
//     a.click();
//     document.body.removeChild(a);
//     window.URL.revokeObjectURL(url);
//      // Remove the downloading icon after download is complete
//      downloadButton.classList.remove('downloading');
// });

// function isValidCredentials(username, password) {
//     // Perform validation here
//     // Username must contain only alphabets
//     const usernamePattern = /^[A-Za-z]+$/;
//     // For simplicity, let's assume any non-empty username and password is valid
//     return usernamePattern.test(username) && username !== '' && password !== '';
// }

// 
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loginError = document.getElementById('loginError');

    loginForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const username = loginForm.querySelector('#username').value.trim();
        const password = loginForm.querySelector('#password').value.trim();
        
        // Validate username and password
        if (!isValidCredentials(username, password)) {
            loginError.textContent = 'Invalid username or password.';
            return;
        } else {
            loginError.textContent = ''; // Clear previous error messages
        }

        // Send login data to server for authentication
        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                password: password,
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Login successful!');
                window.location.href = 'upload_pdf';
            } else {
                loginError.textContent = 'Authentication failed. ' + data.message;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loginError.textContent = 'An error occurred during authentication.';
        });
    });

    function isValidCredentials(username, password) {
        // Perform validation here
        // Username must contain only alphabets
        const usernamePattern = /^[A-Za-z0-9]+$/;
        // For simplicity, let's assume any non-empty username and password is valid
        return usernamePattern.test(username) && username !== '' && password !== '';
    }
});
