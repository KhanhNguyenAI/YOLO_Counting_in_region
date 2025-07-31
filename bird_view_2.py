from ultralytics import YOLO
from ultralytics import solutions
import cv2
import cvzone
import numpy as np 
import math
import sort
model = YOLO("./model/VisDroneModel.pt")

cap = cv2.VideoCapture('./video/bird_view_2.mp4')  # video
assert cap.isOpened(), "Error reading video file"
# Pass region as list
region_points =   np.array([[192, 504], [238, 683], [372, 656], [292, 499]])
regioncounter = solutions.RegionCounter(
    show= False,  # display the frame
    region=region_points,  # pass region points
    model="./model/VisDroneModel.pt",  # model for counting in regions i.e yolo11s.pt
    show_labels = False,
    show_conf = False,

)
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.2)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('./outputVideo/bird_view_2.mp4', fourcc, fps, (506,921))

class_name = {
    '0': 'pedestrian',
    '1': 'people',
    '2': 'bicycle',
    '3': 'car',
    '4': 'van',
    '5': 'truck',
    '6': 'tricycle',
    '7': 'awning-tricycle',
    '8': 'bus',
    '9': 'motor'
}



limits = [174, 761, 210, 902]
limits_left = [316, 689, 388, 676]
total_Count_right = []
total_Count_left = []
limits_top = [207, 543,312, 535]
limits_down = [248, 672, 383, 658]
mask = cv2.imread('./img/r.png')
while True:
    ret, frame = cap.read()
    #logo
    frame = cv2.resize(frame,(506,921))
    img_graphic = cv2.imread('./img/logo.png',cv2.IMREAD_UNCHANGED)
    img_graphic = cv2.cvtColor(img_graphic, cv2.COLOR_BGR2BGRA)
    img_graphic =cv2.resize(img_graphic,(100,100))
    cv2.putText(frame, text='Regions', org=(0, 130),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0, 255, 0), thickness=2)
    frame = cvzone.overlayPNG(frame,img_graphic,(0,0))

    if not ret:
        break
    #mask
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    img_region = cv2.bitwise_and(frame, mask)
    detections = np.empty((0, 5))
#regions

    results2= regioncounter.process(img_region)
    for k, v in results2.region_counts.items():
        cv2.putText(frame, f'{v}', (131, 130), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), thickness=2)

    results = model(img_region, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]
            conf_int = math.ceil((box.conf[0] * 100)) / 100
            cls = box.cls[0]
            class_index = str(int(cls))
            cls_name = class_name[class_index]
            if conf_int > 0:
                # cvzone.putTextRect(frame, f'{cls_name} {conf_int} ', (max(0, x1), max(40, y1)), scale=0.5, thickness=1)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=2, t=1, rt=5, colorR=(255, 0, 255), colorC=(0, 255, 0))
                cx, cy = x1 + w // 2, y1 + h // 2
                # cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                current_array = np.array([x1, y1, x2, y2, conf_int])
                detections = np.vstack((detections, current_array))
    # Chuyển đổi region_points thành mảng NumPy cho cv2.pointPolygonTest
    pts_region = np.array(region_points, np.int32)
    pts_region = pts_region.reshape((-1, 1, 2))
    result_tracker = tracker.update(detections)
    pts_region_display = region_points.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_region_display], isClosed=True, color=(0, 255, 255), thickness=2) # Yellow color for visibility
    for result in result_tracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1


        #main line

        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        cv2.line(frame, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (0, 0, 255), 5)

        # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255), colorC=(0, 255, 0),t=1)
        # cvzone.putTextRect(frame, f'{int(Id)} ', (max(0, x1), max(35, y1)), scale=3, thickness=1, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        color_detected = (0, 0, 255)

        if (limits[0] < cx < limits[2]) and (limits[1] - 5 < cy < limits[3] + 5):
            color_detected = (0, 255, 0)
            if total_Count_right.count(Id) == 0 : 
                total_Count_right.append(Id)
                

                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                cv2.putText(frame, 'Detected', (limits[0], limits[1]), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 255, 0), thickness=2)
            

        
        if (limits_left[0] < cx < limits_left[2]) and (limits_left[1] - 10 < cy < limits_left[3] + 10):
            color_detected = (0, 255, 0)
            if total_Count_left.count(Id) == 0 : 
                total_Count_left.append(Id)
                
                cv2.line(frame, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (0, 0, 255), 5)
                cv2.putText(frame, 'Detected', (limits_left[0],limits_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(0, 255, 0), thickness=2)


        cv2.putText(frame, 'Detected', (0, 178), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=color_detected, thickness=2)
        cv2.putText(frame,f'{len(total_Count_right)} ', (196, 816   ), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), thickness=2)
        cv2.putText(frame,f'{len(total_Count_left)} ', (384, 729), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), thickness=2)
        cv2.putText(frame, f'{len(total_Count_left) + len(total_Count_right)} ', (150, 178), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,255),
                    thickness=2)
        # cvzone.putTextRect(frame, f'{total_Count_right} ', (779, 656), scale=3, thickness=2, offset=10)

    out.write(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('region', img_region)
    cv2.imwrite('./img/bird_view_1.png', frame) #save img test
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()