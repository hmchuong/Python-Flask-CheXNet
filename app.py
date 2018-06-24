import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, json
from xray_classifier.predict import ChoosingMethodModel

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dir_path = os.path.dirname(os.path.realpath(__file__))

choosing_method_model = ChoosingMethodModel()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            prob = choosing_method_model.predict(filename)
            if (prob > 0.5):
                # BSE
                pass
            response = app.response_class(
                response=json.dumps({'should_bse': prob}),
                status=200,
                mimetype='application/json'
            )
            return response
            #return redirect(url_for('uploaded_file',
                                    #filename=filename))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
