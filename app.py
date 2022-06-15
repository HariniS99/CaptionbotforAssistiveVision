from flask import Flask, render_template, url_for, request, redirect
from caption_generator import *
from vgg_intermediate import *
import warnings
from whitenoise import WhiteNoise
from gtts import gTTS

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
# YANDEX_API_KEY = 'YOUR API KEY HERE'
# SECRET_KEY = '7d441f27d441f27567d441f2b6176a'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result_new = None
    fma_dict = None
    if request.method == 'POST':
        img = request.files['image']

        # print(img)
        # print(img.filename)

        img.save("static/" + img.filename)

        caption = test_images('./static/' + img.filename)
        get_fmap("./static/" + img.filename)
        text = caption
        language = 'en'
        speech = gTTS(text=text, lang=language, slow=False)
        filesource = './static/audio/' + img.filename[:5] + '.mp3'
        speech.save(filesource)
        result_new = {
            'image': "static/" + img.filename,
            'description': caption,
            'soundit': './static/audio/' + img.filename[:5] + '.mp3',
        }
        #fmap(img.filename)

    return render_template('index.html', results=result_new)


@app.route('/fmap', methods=['GET', 'POST'])
def fmap():
    fmap_dict = {
        'blk_1':"./static/fmap/blk_2.jpg",
        'blk_2': "./static/fmap/blk_5.jpg",
        'blk_3': "./static/fmap/blk_9.jpg",
        'blk_4': "./static/fmap/blk_13.jpg",
        'blk_5':"./static/fmap/blk_17.jpg",
    }

    return render_template("f_map.html",fmap=fmap_dict)


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
