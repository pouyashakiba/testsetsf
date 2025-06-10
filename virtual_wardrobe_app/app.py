from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import shutil
# Assuming app.py is in virtual_wardrobe_app, preprocessing_utils is in the same directory
import preprocessing_utils

app = Flask(__name__)
# UPLOAD_FOLDER is relative to app.py (which is inside virtual_wardrobe_app)
# So, files will be stored in virtual_wardrobe_app/uploads/
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CKPT_DIR = 'ckpt/'  # Placeholder for checkpoints directory

def run_idm_vton_inference(person_image_path, cloth_image_path, human_parsing_mask_path, densepose_map_path, cloth_mask_path, openpose_keypoints_path, upload_folder):
    """
    Placeholder for actual IDM-VTON inference.
    Currently copies the person image as the 'generated' output.
    """
    print(f"Simulating IDM-VTON inference with person: {person_image_path}, cloth: {cloth_image_path}")
    print(f"  Human parsing mask: {human_parsing_mask_path}")
    print(f"  DensePose map: {densepose_map_path}")
    print(f"  Cloth mask: {cloth_mask_path}")
    print(f"  OpenPose keypoints: {openpose_keypoints_path}")

    # Simulate output - copy person image to a new name
    output_filename = "final_generated_output.png"
    # Output path should be in the main uploads folder, not the preprocessed_inputs subfolder
    output_path = os.path.join(upload_folder, output_filename)
    try:
        # Ensure the source person_image_path is correct (it should be from the main uploads folder)
        shutil.copy(person_image_path, output_path)
        print(f"Dummy output generated at: {output_path}")
        return output_filename
    except Exception as e:
        print(f"Error in dummy inference (copying file): {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    generated_image_name = request.args.get('generated_image')
    return render_template('index.html', generated_image_name=generated_image_name)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'person_image' not in request.files or \
       'top_image' not in request.files: # bottom_image is optional for VTON
        return "Error: Missing person or top image files", 400

    person_image_file = request.files['person_image']
    top_image_file = request.files['top_image']
    # bottom_image_file is still present in the form but not used in this VTON flow
    # bottom_image_file = request.files.get('bottom_image')

    if person_image_file.filename == '' or top_image_file.filename == '':
        return "Error: No selected file for person or top image", 400

    if person_image_file and top_image_file:
        person_filename = secure_filename(person_image_file.filename)
        cloth_filename = secure_filename(top_image_file.filename) # Using top_image as cloth

        # Save original uploaded files directly into UPLOAD_FOLDER
        # app.root_path is 'virtual_wardrobe_app'
        # app.config['UPLOAD_FOLDER'] is 'uploads/'
        # So, person_saved_path will be 'virtual_wardrobe_app/uploads/person_image.jpg'
        person_saved_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], person_filename)
        cloth_saved_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], cloth_filename)

        # Ensure the main upload directory exists (already done at app init, but good practice)
        os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

        person_image_file.save(person_saved_path)
        top_image_file.save(cloth_saved_path)

        # Define output directory for preprocessed files
        # This will be 'virtual_wardrobe_app/uploads/preprocessed_inputs'
        output_preprocess_dir = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "preprocessed_inputs")
        os.makedirs(output_preprocess_dir, exist_ok=True)

        # Call preprocessing functions
        # These functions expect full paths for input images and output directory
        human_mask_path = preprocessing_utils.generate_human_parsing_mask(person_saved_path, output_preprocess_dir, CKPT_DIR)
        densepose_path = preprocessing_utils.generate_densepose_map(person_saved_path, output_preprocess_dir, CKPT_DIR)
        cloth_mask_path = preprocessing_utils.generate_cloth_mask(cloth_saved_path, output_preprocess_dir) # No CKPT_DIR for this dummy
        openpose_path = preprocessing_utils.generate_openpose_keypoints(person_saved_path, output_preprocess_dir, CKPT_DIR)

        if not all([human_mask_path, densepose_path, cloth_mask_path, openpose_path]):
            print("A preprocessing step failed.")
            # Potentially flash a message to the user
            return redirect(url_for('index')) # Or an error page

        # Call the new inference placeholder
        # The inference function needs paths to original images and preprocessed files.
        # It also needs the main UPLOAD_FOLDER to save its final output.
        # The paths returned by preprocessing are already full paths.
        generated_image_filename = run_idm_vton_inference(
            person_saved_path,
            cloth_saved_path,
            human_mask_path,
            densepose_path,
            cloth_mask_path,
            openpose_path,
            os.path.join(app.root_path, app.config['UPLOAD_FOLDER']) # Pass full path to upload_folder
        )

        if generated_image_filename:
            return redirect(url_for('index', generated_image=generated_image_filename))
        else:
            print("Inference failed.")
            # Potentially flash a message
            return redirect(url_for('index'))


    return "Error: File upload failed for an unknown reason", 500

@app.route('/serve_upload/<filename>')
def serve_upload(filename):
    # This serves files from the main UPLOAD_FOLDER (e.g., virtual_wardrobe_app/uploads/)
    return send_from_directory(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), filename)

if __name__ == '__main__':
    app.run(debug=True)
