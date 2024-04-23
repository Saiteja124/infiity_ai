// document.getElementById('fieldsForm').addEventListener('submit', function(event) {
//     event.preventDefault();
//     var fields = document.getElementById('fields').value;
//     document.getElementById('processingMessage').style.display = 'block'; // Show processing message
//     extractFields(fields);
// });

// function extractFields(fields) {
//     var formData = new FormData();
//     formData.append('fields', fields);

//     fetch('/process_specific_fields', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error('Network response was not ok');
//         }
//         return response.blob();
//     })
//     .then(blob => {
//         // Clear the result message
//         // document.getElementById('result').textContent = '';

//         // Show the completion message after a short delay
//         // setTimeout(function() {
//         //     document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
//         // }, 500); // Delay of 500 milliseconds (adjust as needed)
//         // Create a download link for the returned CSV file
//         var url = window.URL.createObjectURL(blob);
//         var a = document.createElement('a');
//         a.href = url;
//         a.download = 'extracted_columns.csv';
//         document.body.appendChild(a);
//         a.click();
//         document.body.removeChild(a);
//         window.URL.revokeObjectURL(url);
//         setTimeout(function() {
//             document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
//             setTimeout(function() {
//                 document.getElementById('result').textContent = ''; // Clear the message after 3 seconds
//             }, 3000);
//         }, 100); // Show the message after a slight delay for smoother transition

//         // setTimeout(function() {
//         //     document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
//         // }, 3000);
//         // document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
//         // document.getElementById('result').textContent = ''; // Clear the result message
//     })
//     .catch(error => {
//         console.error('There was a problem with the fetch operation:', error);
//         document.getElementById('result').textContent = 'Error: ' + error.message;
//     })
//     .finally(() => {
//     document.getElementById('processingMessage').style.display = 'none'; // Hide processing message
//     document.getElementById('result').textContent = ''; // Clear the result message
// });
// }

document.getElementById('fieldsForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var checkedItems = document.querySelectorAll(".item.checked");
    var fields = Array.from(checkedItems).map(item => item.querySelector(".item-text").textContent).join(',');
    // var fields = ''; // Extract the selected fields here
    // var fields = document.getElementById('fields').value;
    document.getElementById('processingMessage').style.display = 'block'; // Show processing message
    extractFields(fields);
});

function extractFields(fields) {
    var formData = new FormData();
    formData.append('fields', fields);

    fetch('/process_specific_fields', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        // Clear the result message
        // document.getElementById('result').textContent = '';

        // Show the completion message after a short delay
        // setTimeout(function() {
        //     document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
        // }, 500); // Delay of 500 milliseconds (adjust as needed)
        // Create a download link for the returned CSV file
        var url = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'extracted_columns.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        setTimeout(function() {
            document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
            setTimeout(function() {
                document.getElementById('result').textContent = ''; // Clear the message after 3 seconds
            }, 3000);
        }, 100); // Show the message after a slight delay for smoother transition

        // setTimeout(function() {
        //     document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
        // }, 3000);
        // document.getElementById('result').textContent = 'Extraction complete. Downloading CSV file...';
        // document.getElementById('result').textContent = ''; // Clear the result message
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        document.getElementById('result').textContent = 'Error: ' + error.message;
    })
    .finally(() => {
    document.getElementById('processingMessage').style.display = 'none'; // Hide processing message
    document.getElementById('result').textContent = ''; // Clear the result message
});
}