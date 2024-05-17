from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from PIL import Image
import pytesseract
from tensorflow import keras # Import Keras from TensorFlow
# from keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

import regex as re
import nltk
import numpy as np

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        text = evaluate(file_path)
        return render_template('index.html', form=form, text=text)  # Pass text to the template
    return render_template('index.html', form=form, text=None)

def evaluate(file_path):
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    prediction = prediction_on_custom_input(text)
    
    return prediction
    
def prediction_on_custom_input(text):
    encoded = word_embedding(text)
    max_length = 42
    # Print the shape of encoded to debug
    print("Encoded shape:", encoded.shape)

    # Assuming encoded has shape (batch_size, timesteps, features)
    # Modify if needed based on the actual shape
    padded_encoded_title = keras.preprocessing.sequence.pad_sequences(encoded, maxlen=max_length, padding='pre')
    # Print the shape of padded_encoded_title to debug
    print("Padded Encoded shape:", padded_encoded_title.shape)

    model_path = 'Model/-Fake_news_predictor.h5'
    custom_objects = {'Orthogonal': keras.initializers.Orthogonal(), 'LSTM': keras.layers.LSTM, 'GlorotUniform': keras.initializers.GlorotUniform()}

    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    output = model.predict(padded_encoded_title)
    output = np.where(0.4 > output, 1, 0)
    if output[0][0] == 1:
        return 'Yes this News is fake'
    return 'No, It is not fake'


def word_embedding(text):
    preprocessed_text = preprocess_filter(text)
    return one_hot_encoded(preprocessed_text)


def one_hot_encoded(text, vocab_size=5000, max_length=40):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([text])
    hot_encoded = tokenizer.texts_to_sequences([text])
    hot_encoded = keras.preprocessing.sequence.pad_sequences(hot_encoded, maxlen=max_length, padding='pre')
    return hot_encoded



text_cleaning = "\b0\S*|\b[^A-Za-z0-9]+"

def preprocess_filter(text, stem=False):
    text = re.sub(text_cleaning, " ", str(text.lower()).strip())
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
                token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)

if __name__ == '__main__':
    app.run(debug=True)
