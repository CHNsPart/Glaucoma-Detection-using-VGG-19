from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
'''
dic = {
    0: "Glaucoma",
    1: "Non-Glaucoma"
}
'''

model = load_model("vgg19forapp.h5")


def predict_label(img_path):

    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)
    i = i.reshape(1, 224, 224, 3)
    p = model.predict(i)
    #result = [p[0][0]*100]
    # print(result)
    return p


# routes
@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "CHNsPart Project"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)
        acc = [p[0][0]*100]
        loss = [p[0][1]*100]
        title = ""
        if acc[0] > 50:
            title = "Glaucoma 😍"
        elif acc[0] < 50:
            title = "Not Glaucoma 😔"
        else:
            title = "Not Sure 🤔"

    return render_template("index.html", prediction=acc, loss=loss, title=title, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
