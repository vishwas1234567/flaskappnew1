

from flask import Flask, render_template, Response
import cv2
app=Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            car_cascade = cv2.CascadeClassifier("Haarcascades/car.xml")
            smile_cascade = cv2.CascadeClassifier('Haarcascades/smile.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each feature
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                    # for (body_x,body_y,body_w,body_h) in body:
                    #     cv2.rectangle(roi_color, (body_x,body_y), (body_x+body_w, body_y+body_h), (0, 0, 255), 2)
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 1)
                        car = car_cascade.detectMultiScale(roi_gray, 2.0, 20)
                        for (car_x,car_y,car_w,car_h) in car:
                            cv2.rectangle(roi_color, (car_x,car_y), (car_x+car_w, car_y+car_h), (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)