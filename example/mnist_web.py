import sys
sys.path.insert(1, '..')
import numpy as np
from braingrad.optim import SGD
from braingrad.engine import Tensor
from mnist import BrainNet, train, X_test
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

app = Flask(__name__)

model = BrainNet()
optim = SGD([model.l1, model.l2])

def convert_img(img):
    """
    Args:
        img (ndarray)
    Returns:
        uri (string) : base64 encoding of the image
    """    
    from base64 import b64encode        
    from io import BytesIO
    
    img = Image.fromarray(img)
    raw_img = BytesIO()
    img.save(raw_img, "JPEG")
    raw_img.seek(0)
    img_base64 = b64encode(raw_img.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)

    return uri

@app.route('/')
def home():
    return redirect(url_for('random'))

@app.route('/random')
def random():
    samp = np.random.randint(X_test.shape[0])
    img = Tensor(X_test[samp].reshape((-1, 28*28)))
    
    pred = model.forward(img)
    pred = np.argmax(pred.data, axis=1)

    return render_template("random.html", img=convert_img(X_test[samp]), number=str(pred[0]))


if __name__ == '__main__':

    train(model, optim, BS=128)
    app.run(host='0.0.0.0', debug=True)
