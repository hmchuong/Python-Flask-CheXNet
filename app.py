import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, json
from xray_classifier.predict import ChoosingMethodModel
from bse_clahe.model import AELikeModel
from chexnet.HeatmapGenerator import HeatmapGenerator

UPLOAD_FOLDER = 'images/upload'
BSE_FOLDER = 'images/bse'
HEATMAP_FOLDER = 'images/heatmap'
ALLOWED_EXTENSIONS = set(['png'])
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BSE_FOLDER'] = BSE_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
dir_path = os.path.dirname(os.path.realpath(__file__))

choosing_method_model = ChoosingMethodModel()
bse_clahe = AELikeModel(1024, 0.84, trained_model=os.path.join(dir_path, 'bse_clahe/model/model'))
chexnet_origin = HeatmapGenerator(os.path.join(dir_path, "chexnet/models/Origin.pth.tar"), 'DENSE-NET-121', 14, 224)
chexnet_bse = HeatmapGenerator(os.path.join(dir_path, "chexnet/models/CLAHE-0152.pth.tar"), 'DENSE-NET-121', 14, 224)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/origin/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/bse/<filename>')
def bse_file(filename):
    return send_from_directory(app.config['BSE_FOLDER'],
                               filename)

@app.route('/heatmap/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'],
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
            origin_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(origin_filename)
            prob = choosing_method_model.predict(origin_filename)
            bse_url = url_for('uploaded_file',filename=filename)
            heatmap_filename = os.path.join(app.config['HEATMAP_FOLDER'], filename)
            results = {}
            if (prob > 0.5):
                #BSE
                bse_filename = os.path.join(app.config['BSE_FOLDER'], filename)
                bse_clahe.test(origin_filename, bse_filename, need_invert=True)
                bse_url = url_for('bse_file',filename=filename)
                predictions = chexnet_bse.generate(bse_filename, heatmap_filename, 224)
            else:
                predictions = chexnet_origin.generate(origin_filename, heatmap_filename, 224)
            print(predictions)
            for i in range(len(CLASS_NAMES)):
                results[CLASS_NAMES[i]] = float(predictions[0,i])
            response = app.response_class(
                response=json.dumps({'should_bse': prob,
                                     'origin': url_for('uploaded_file',filename=filename),
                                     'bse': bse_url,
                                     'heatmap': url_for('heatmap_file',filename=filename),
                                     'results': results}),
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
