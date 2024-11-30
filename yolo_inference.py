#Importing packages
from ultralytics import YOLO

#Loading the model
#model=YOLO('yolov8x')
model=YOLO('models/best.pt')

#Running the YOLO model
results = model.predict('input_videos/08fd33_4.mp4',save=True)

#Show the results
print(results[0])

print("#########################################################################")
for box in results[0].boxes:
    print(box)

