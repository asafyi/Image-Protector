from flask import Flask, request, render_template, send_file
from flask_cors import CORS
from PIL import Image
from attack import * 
from zipfile import ZipFile
import requests
import io

# discord bot notifications
url = "https://discordapp.com/api/webhooks/1114490843680743495/HMOSRSL6tSPl9Juydz_5j2dnlP3WMRhYtnH3F8A5VADZDTTg4wL8zKS3wUTNGnsUE3WS"
loading_data = {
    "username": "gcp_script_bot",
    "content": "Loading Website..."
    }

online_data = {
    "username": "gcp_script_bot",
    "content": "Website Online!"
    }

requests.post(url, json=loading_data) # alert discord that loading the website
DIF_model, GAN_net = None, None
app = Flask(__name__, template_folder="dist", static_folder="dist/assets")
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
DIF_model, GAN_net = init_models() # loading diffusion and StyleGan models
requests.post(url, json=online_data) # alert discord that the website is online


# loading the website main page
@app.route('/', methods= ['GET', 'POST'])
def get_message():
    print("Got request in main function")
    return render_template("index.html")


# handle request of images upload and send the protected images back to the client
@app.route('/upload', methods=['POST'])
def upload_static_file():
    print("Got request in static files")
    print(request.files.getlist('files'))
    file_bytes, format, name = files_handler(request.files.getlist('files'))
    print(f"file name: {name} , {format}") # it doesn't get here
    return send_file(file_bytes, mimetype=format, as_attachment=True, download_name=name) 


def files_handler(files):
    """
    the function get the list of files and send every image as PIL image to the run_attack function.
    the function check thenumber of files uploaded in order to decide the file's format that will be returned
    """
    if len(files) == 1:
        print("got 1 file!")
        f = files[0]
        img = Image.open(f.stream)
        img = run_attack(img, DIF_model, GAN_net)
        img_bytes = io.BytesIO()
        format = f.content_type
        print("======" + format)
        img.save(img_bytes, format=format[6:])
        img_bytes.seek(0)
        return img_bytes, format, f.name
    else:
        if len(files) == 0:
            print("got 0 files!")
            return
        
        print("more than 1 file!")

        results = []
        for f in files:
            print(f)
            img = Image.open(f.stream)
            img = run_attack(img, DIF_model, GAN_net)
            img_bytes = io.BytesIO()
            format = f.content_type
            print("======" + f.filename)
            img.save(img_bytes, format=format[6:])
            img_bytes.seek(0)
            results.append((img_bytes, f.filename))
            
        # Create an in-memory zip file from the in-memory image file data.
        zip_file_bytes_io = io.BytesIO()

        with ZipFile(zip_file_bytes_io, 'w') as zip_file:
            for bytes_stream, image_name in results:
                zip_file.writestr(image_name, bytes_stream.getvalue())
        zip_file_bytes_io.seek(0)
        return zip_file_bytes_io, "application/zip", "all.zip"

