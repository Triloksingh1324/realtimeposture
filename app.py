from flask import Flask, render_template, request
from video_stream import VideoStream

app = Flask(__name__)

# Initialize the VideoStream object
video_stream = VideoStream()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        choice = request.form['choice']
        if choice == '1':
            video_stream.set_classifier('pushup')
        elif choice == '2':
            video_stream.set_classifier('bicep')
        elif choice == '3':
            video_stream.set_classifier('plank')
        elif choice == '4':
            video_stream.set_classifier('Tree')
        elif choice == '5':
            video_stream.set_classifier('TPose')
        elif choice == '6':
            video_stream.set_classifier('WarriorPose')

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return video_stream.stream()

if __name__ == '__main__':
    app.run(debug=True,port=5000)
