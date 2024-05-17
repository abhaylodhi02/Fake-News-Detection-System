function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            document.getElementById('previewContainer').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const file = document.getElementById('imageInput').files[0];
    if (file) {
        // Handle file upload using AJAX or other methods
        console.log('File uploaded:', file);
        alert('File uploaded successfully!');
        // You can add code here to handle the file upload to a server
    } else {
        alert('Please select an image to upload.');
    }
});
