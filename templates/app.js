document.addEventListener('DOMContentLoaded', function() {
    var captureImageBtn = document.getElementById('captureImageBtn');
    var pickImageBtn = document.getElementById('pickImageBtn');
    var scanImageBtn = document.getElementById('scanImageBtn');
    var imagePreview = document.getElementById('imagePreview');
    var progressBar = document.getElementById('progressBar');
    var scanProgress = document.getElementById('scanProgress');
    var progressText = document.getElementById('progressText');
    var resultContainer = document.getElementById('result');

    var scannedImageDataURL;

    // Event listener for capturing an image
    captureImageBtn.addEventListener('click', function() {
        // ... (same as before)
    });

    // Event listener for picking an image from a file
    pickImageBtn.addEventListener('click', function() {
        var input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';

        input.addEventListener('change', function(event) {
            var file = event.target.files[0];

            if (file) {
                var reader = new FileReader();

                reader.onload = function(e) {
                    // Display the picked image
                    displayImage(e.target.result);
                };

                reader.readAsDataURL(file);
            }
        });

        // Trigger the file input click
        input.click();
    });

    // Event listener for scanning the picked image
    scanImageBtn.addEventListener('click', function() {
        if (!scannedImageDataURL) {
            alert('Please capture or pick an image first.');
            return;
        }

        // Display progress bar and hide result
        progressBar.style.display = 'block';
        resultContainer.innerHTML = '';

        // Simulate scanning process with a timer
        var progress = 0;
        var scanInterval = setInterval(function() {
            progress += 10;
            scanProgress.value = progress;
            progressText.textContent = progress + '% Scanned...';

            if (progress >= 100) {
                clearInterval(scanInterval);

                // Simulate sending the scanned image to the backend
                sendImageToBackend(scannedImageDataURL);
            }
        }, 500);
    });

    // Function to send the image to the backend
    function sendImageToBackend(imageDataURL) {
        // Simulate AJAX request to the backend (replace with actual implementation)
        setTimeout(function() {
            // Simulated backend response
            var backendResponse = Math.random() < 0.5 ? 'Fresh' : 'Spoiled';

            // Hide progress bar
            progressBar.style.display = 'none';

            // Display backend response
            resultContainer.innerHTML = '<p>Result: ' + backendResponse + '</p>';
        }, 2000);
    }

    // Function to display the image in the preview div
    function displayImage(imageDataURL) {
        scannedImageDataURL = imageDataURL;

        // Create an image element
        var image = document.createElement('img');
        image.src = imageDataURL;
        image.alt = 'Captured/Picked Image';

        // Clear previous content and append the new image
        imagePreview.innerHTML = '';
        imagePreview.appendChild(image);
    }
});