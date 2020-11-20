import os, cv2, dlib, torch
import datetime
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import pickle as pkl
import numpy as np
from simple_pose_estimator import Model
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_directory = './models'
model_number = 92500
model_file = "epoch_" + str(model_number) + ".pt"
mod_path = os.path.join(model_directory,model_file)

image_output_directory = './images'
model_pic = 'data/a_girl.jpg'
today = str(datetime.date.today())
output_image_path = os.path.join(image_output_directory,today+'.png')

model = Model()
checkpoint = torch.load(mod_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_model/shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []
    
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points
        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)


im = cv2.imread(model_pic, cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
face_points = detect_face_points(im)

for x, y in face_points:
    cv2.circle(im, (x, y), 1, (0, 255, 0), -1)

x, _ = pkl.load(open('data/samples.pkl', 'rb'))
std = StandardScaler()
std.fit(x)
features = compute_features(face_points)
features = std.transform(features)
features = torch.FloatTensor(np.ndarray.tolist(features))

y_pred = model(features)

roll_pred, pitch_pred, yaw_pred = y_pred[0].tolist()
print(' Roll: {:.2f}°'.format(roll_pred))
print('Pitch: {:.2f}°'.format(pitch_pred))
print('  Yaw: {:.2f}°'.format(yaw_pred))

imsave(output_image_path,im)
