from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
# UPLOAD_FOLDER is relative to app.py (which is inside virtual_wardrobe_app)
# So, files will be stored in virtual_wardrobe_app/uploads/
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def mock_ai_processor(person_image_path, top_image_path, bottom_image_path):
    # Simulate AI processing by copying the person's image to a new file
    generated_image_name = "generated_image.png"
    # person_image_path is the full path to the uploaded person image
    # e.g. virtual_wardrobe_app/uploads/person.jpg
    # app.config['UPLOAD_FOLDER'] is 'uploads/'
    # To get the correct output path for generated_image.png inside virtual_wardrobe_app/uploads/
    # we should join app.root_path (which is virtual_wardrobe_app) with app.config['UPLOAD_FOLDER']
    # or simply use the dirname of person_image_path as done previously.
    upload_folder_full_path = os.path.dirname(person_image_path)
    generated_image_path = os.path.join(upload_folder_full_path, generated_image_name)
    shutil.copy(person_image_path, generated_image_path)
    return generated_image_name # Return only the filename for serving

@app.route('/', methods=['GET'])
def index():
    generated_image_name = request.args.get('generated_image')
    return render_template('index.html', generated_image_name=generated_image_name)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'person_image' not in request.files or \
       'top_image' not in request.files or \
       'bottom_image' not in request.files:
        return "Error: Missing one or more image files", 400

    person_image_file = request.files['person_image']
    top_image_file = request.files['top_image']
    bottom_image_file = request.files['bottom_image']

    if person_image_file.filename == '' or \
       top_image_file.filename == '' or \
       bottom_image_file.filename == '':
        return "Error: No selected file for one or more images", 400

    if person_image_file and top_image_file and bottom_image_file:
        person_filename = secure_filename(person_image_file.filename)
        top_filename = secure_filename(top_image_file.filename)
        bottom_filename = secure_filename(bottom_image_file.filename)

        # Construct full paths for saving and processing
        # app.root_path is the path to the directory where app.py is (virtual_wardrobe_app)
        person_image_filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], person_filename)
        top_image_filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], top_filename)
        bottom_image_filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], bottom_filename)

        os.makedirs(os.path.dirname(person_image_filepath), exist_ok=True) # ensure dir exists

        person_image_file.save(person_image_filepath)
        top_image_file.save(top_image_filepath)
        bottom_image_file.save(bottom_image_filepath)

        generated_image_filename = mock_ai_processor(person_image_filepath, top_image_filepath, bottom_image_filepath)

        return redirect(url_for('index', generated_image=generated_image_filename))

    return "Error: File upload failed for an unknown reason", 500

@app.route('/serve_upload/<filename>')
def serve_upload(filename):
    # send_from_directory needs the directory path relative to the application root if UPLOAD_FOLDER is relative,
    # or an absolute path.
    # app.config['UPLOAD_FOLDER'] is 'uploads/'
    # The directory for send_from_directory should be os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    # However, Flask's send_from_directory can also take a directory relative to app.root_path if the path doesn't start with '/'
    # So, app.config['UPLOAD_FOLDER'] ('uploads/') should work directly if it's interpreted as relative to app.root_path.
    # Let's be explicit for clarity and robustness:
    return send_from_directory(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), filename)

if __name__ == '__main__':
    app.run(debug=True)
