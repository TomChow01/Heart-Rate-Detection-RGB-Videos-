import os
import cv2
from test_model import *
import argparse
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory, Response


UPLOAD_FOLDER = r'G:\ML Projects\ISI\Baby Monitoring\Heart Rate Detection\Deploy\hr_demo\uploads'
ALLOWED_EXTENSIONS = {'jpg', 'mp4', 'avi'}

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
           
           
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # If video path is given    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files.get('file')
        #For Webcam/IP Cam capture
        if file.filename == '':
            video_path = None
            flash('No selected file')
            # return redirect(request.url)
            return redirect(url_for('display_hr_webcam'))
        
        filename = file.filename
        print("Filename: ", os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        


        
        
        
        return redirect(url_for('display_hr',
                                video_path=video_path))


    return render_template('index.html')

@app.route('/display_hr_webcam')
def display_hr_webcam():
    hr = HeartRate(video_path = None, save = False, color_magnify = False, mask_path = None, out_path = None)
    return Response(hr.run(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/display_hr/<video_path>')
def display_hr(video_path):
    hr = HeartRate(video_path = video_path, save = False, color_magnify = False, mask_path = None, out_path = None)
    return Response(hr.run(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
