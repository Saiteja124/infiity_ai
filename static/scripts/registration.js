// document.addEventListener('DOMContentLoaded', function() {
//     const registrationForm = document.getElementById('registrationForm');

//     registrationForm.addEventListener('submit', function(event) {
//         event.preventDefault();

//         // Check if the form is already being submitted
//         if (registrationForm.getAttribute('data-submitting') === 'true') {
//             return; // Do nothing if already submitting
//         }

//         // Set a flag to indicate the form is being submitted
//         registrationForm.setAttribute('data-submitting', 'true');

//         const username = registrationForm.querySelector('#username').value.trim();
//         const email = registrationForm.querySelector('#email').value.trim(); // Get email value
//         const password = registrationForm.querySelector('#password').value.trim();

//         // Validate username, email, and password
//         if (!isValidUsername(username)) {
//             alert('Please enter a valid username.');
//             registrationForm.removeAttribute('data-submitting'); // Reset the flag
//             return;
//         }

//         if (!isValidEmail(email)) {
//             alert('Please enter a valid email address.');
//             registrationForm.removeAttribute('data-submitting'); // Reset the flag
//             return;
//         } 

//         if (!isValidPassword(password)) {
//             alert('Please enter a valid password. Password must be at least 8 characters long.');
//             registrationForm.removeAttribute('data-submitting'); // Reset the flag
//             return;
//         }

//         // Send registration data to server for authentication
//         fetch('/register', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//                 username: username,
//                 email: email,
//                 password: password,
//             }),
//         })
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error(`HTTP error! Status: ${response.status}`);
//             }
//             return response.json();
//         })
//         .then(data => {
//             if (data.registration_success) {
//                 alert('Registration successful! Redirecting to login page.');
//                 window.location.href = 'login';
//             } else if (data.username_exists) {
//                 alert('Username already exists');
//             } else if (data.email_exists) {
//                 alert('Email already exists');
//             } else {
//                 alert('Registration failed. Please try again.');
//             }
//             // } else {
//             //     alert('Registration failed. ' + data.message);
//             // }
//         })
//         .catch(error => {
//             console.error('Error:', error);
//             alert('An error occurred during registration.');
//         })
//         .finally(() => {
//             registrationForm.removeAttribute('data-submitting'); // Reset the flag regardless of success or failure
//         });
//     });


//     //     // For demonstration purposes, I'll simulate a successful registration
//     //     alert('Registration successful! Redirecting to login page.');
//     //     window.location.href = 'login';
//     // });

//     // Function to validate username
//     function isValidUsername(username) {
//         // Username must not be empty and should contain only alphanumeric characters
//         return /^[a-zA-Z0-9]+$/.test(username);
//     }

//     // Function to validate email
//     function isValidEmail(email) {
//         // Use a regular expression to validate email format
//         return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
//     }

//     // Function to validate password
//     function isValidPassword(password) {
//         // Password must be at least 8 characters long
//         return password.length >= 8;
//     }
// });

document.addEventListener('DOMContentLoaded', function() {
    const registrationForm = document.getElementById('registrationForm');

    registrationForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const usernameInput = document.getElementById('username');
        const emailInput = document.getElementById('email');
        const passwordInput = document.getElementById('password');

        const usernameError = document.getElementById('usernameError');
        const emailError = document.getElementById('emailError');
        const passwordError = document.getElementById('passwordError');

        // Clear previous error messages
        usernameError.textContent = '';
        emailError.textContent = '';
        passwordError.textContent = '';

        const username = usernameInput.value.trim();
        const email = emailInput.value.trim();
        const password = passwordInput.value.trim();

        // Validate username, email, and password
        if (!isValidUsername(username)) {
            usernameError.textContent = 'Please enter a valid username.';
            return;
        }

        if (!isValidEmail(email)) {
            emailError.textContent = 'Please enter a valid email address.';
            return;
        } 

        if (!isValidPassword(password)) {
            passwordError.textContent = 'Please enter a valid password. Password must be at least 8 characters long.';
            return;
        }

        // Send registration data to server for authentication
        fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                email: email,
                password: password,
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('HTTP error! Status: ${response.status}');
            }
            return response.json();
        })
        .then(data => {
            if (data.registration_success) {
                alert('Registration successful! Redirecting to login page.');
                window.location.href = 'login';
            } else if (data.username_exists) {
                usernameError.textContent = 'Username already exists';
            } else if (data.email_exists) {
                emailError.textContent = 'Email already exists';
            } else {
                alert('Registration failed. Please try again.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during registration.');
        });
    });

    // Function to validate username
    function isValidUsername(username) {
        // Username must not be empty and should contain only alphanumeric characters
        return /^[a-zA-Z0-9]+$/.test(username);
    }

    // Function to validate email
    function isValidEmail(email) {
        // Use a regular expression to validate email format
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    // Function to validate password
    function isValidPassword(password) {
        // Password must be at least 8 characters long
        return password.length >= 8;
    }
});