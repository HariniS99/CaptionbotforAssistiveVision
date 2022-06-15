from flask import Flask, render_template, url_for, request, redirect
import werkzeug
import numpy
from werkzeug.utils import secure_filename

from caption_generator import *
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("./static/"+filename)
    return "Image Uploaded Successfully"




    #img = request.files['file']

    #basepath = os.path.dirname(__file__)
    #file_path = os.path.join(basepath, 'images\\res\\', secure_filename(f.filename))
    #f.save(file_path)
    #img.save("static/" + img.filename)

    #caption = test_images('./static/' + img.filename)

    #return str(caption)
    #return "Flask Server & Android are Working Successfully"


app.run(host="0.0.0.0", port=5000, debug=True)
