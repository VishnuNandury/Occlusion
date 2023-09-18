#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
from scipy.spatial import distance
import numpy as np
from collections import OrderedDict
from yo_ut import particle
from imutils.video import FPS

# ##### Object Tracking Class

# In[2]:


class Tracker:
    def __init__(self, maxLost = 30):           # maxLost: maximum object lost counted when the object is being tracked
        self.nextObjectID = 0                   # ID of next object
        self.objects = OrderedDict()            # stores ID:Locations
        self.lost = OrderedDict()               # stores ID:Lost_count
        self.pobject=OrderedDict()
        self.maxLost = maxLost                  # maximum number of frames object was not detected.
        self.clr=OrderedDict()
        
    def addObject(self, new_object_location,loc,img):
        self.objects[self.nextObjectID] = new_object_location    # store new object location
        self.lost[self.nextObjectID] = 0                         # initialize frame_counts for when new object is undetected
        #create particles
        color=list(np.random.randint(256, size=3))
        self.clr[self.nextObjectID]=[int(color[0]),int(color[1]),int(color[2])]
                
        self.pobject[self.nextObjectID]=particle(loc[0],loc[1],loc[2],loc[3],50,self.clr[self.nextObjectID])
        self.pobject[self.nextObjectID].drawParticles(img, color=[0,0,255], radius=2) 
        self.nextObjectID += 1
    
    def removeObject(self, objectID):                          # remove tracker data after object is lost
        del self.objects[objectID]
        del self.lost[objectID]
        #del self.pobject[objectID]
    
    @staticmethod
    def getLocation(bounding_box):
        xlt, ylt, xrb, yrb = bounding_box
        cent=(int((xlt + xrb) / 2.0),int((ylt + yrb) / 2.0))
        
        det=[xlt, ylt, xrb-xlt, yrb-ylt]
        return cent,det
    
    def update(self,img,  detections):
        coin = np.random.uniform()
        noise_prob=0.15
        std=25
        if(coin>=1.0-noise_prob):
         x_n=int(np.random.uniform(-300,300))
         y_n=int(np.random.uniform(-300,300))
        else:
         x_n=0;
         y_n=0;

        if len(detections) == 0:   # if no object detected in the frame
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] +=1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            
            return self.objects
        for objectID in list(self.objects.keys()):
         x_c,y_c=self.objects[objectID][0],self.objects[objectID][1]
         x_c+=x_n;
         y_c+=y_n;         
         #predict position of the target
         self.pobject[objectID].predict(x_velocity=0,y_velocity=0,std=25)
         color=list(np.random.randint(256, size=3))
         clr=[int(color[0]),int(color[1]),int(color[2])]
                
         #Drawing the particles.
         self.pobject[objectID].drawParticles(img,color=clr)
                
         #Estimate the next position using the internal model
         x_estimated, y_estimated, _, _ = self.pobject[objectID].estimate()
         cv.circle(img, (x_estimated, y_estimated), 3,self.clr[objectID], 5)
                 
         #Update the filter with the last measurements
         self.pobject[objectID].update(x_c,y_c)

         #resample
         self.pobject[objectID].resample()






        new_object_locations = np.zeros((len(detections), 2), dtype="int")     # current object locations
        loc= np.zeros((len(detections),4), dtype="int")
        #print ("**********")
        for (i, detection) in enumerate(detections):
         new_object_locations[i],loc[i] = self.getLocation(detection)
         #print(detection)   
        if len(self.objects)==0:
            for i in range(0, len(detections)):
             self.addObject(new_object_locations[i],loc[i],img)
             
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))
            
            D = distance.cdist(previous_object_locations, new_object_locations) # pairwise distance between previous and current
            print("D: ",D)
            row_idx = D.min(axis=1).argsort()   # (minimum distance of previous from current).sort_as_per_index
            print("row_idx",row_idx)
            cols_idx = D.argmin(axis=1)[row_idx]   # index of minimum distance of previous from current
            print("cols_idx",cols_idx) 
            assignedRows, assignedCols = set(), set()
            
            for (row, col) in zip(row_idx, cols_idx):
                
                if row in assignedRows or col in assignedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = new_object_locations[col]
                self.lost[objectID] = 0
                #print(self.objects[objectID])
                

                x_c,y_c=self.objects[objectID][0],self.objects[objectID][1]
                x_c+=x_n;
                y_c+=y_n;
                #predict position of the target
                self.pobject[objectID].predict(x_velocity=5,y_velocity=2,std=25)
                
                #Drawing the particles.
                self.pobject[objectID].drawParticles(img)
                
                #Estimate the next position using the internal model
                x_estimated, y_estimated, _, _ = self.pobject[objectID].estimate()
                cv.circle(img, (x_estimated, y_estimated), 3, [0,255,0], 5)
                 
                #Update the filter with the last measurements
                self.pobject[objectID].update(x_c,y_c)

                #resample
                self.pobject[objectID].resample()
                




                assignedRows.add(row)
                assignedCols.add(col)
            print("assigned Rows :",assignedRows )   
            print("assigned cols :",assignedCols )   
            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)
            print("unassigned Rows :",unassignedRows )   
            print("unassigned cols :",unassignedCols ) 
            print("**********************")
            if D.shape[0]>=D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1
                    
                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)
                        
            else:
                for col in unassignedCols:
                    #print("NOL:",new_object_locations[col])
                    self.addObject(new_object_locations[col],loc[col],img)
            
        return self.objects


# #### Loading Object Detector Model

# ##### YOLO Object Detection and Tracking
# 
# Here, the YOLO Object Detection Model is used.



yolomodel = {"config_path":"./yolov3.cfg",
              "model_weights_path":"./yolov3.weights",
              "coco_names":"./coco.names",
              "confidence_threshold": 0.5,
              "threshold":0.3
             }

net = cv.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])
labels = open(yolomodel["coco_names"]).read().strip().split("\n")


# In[4]:


np.random.seed(12345)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
print(layer_names)

bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))


# ##### Instantiate the Tracker Class

# In[5]:


maxLost = 30  # maximum number of object losts counted when the object is being tracked
tracker = Tracker(maxLost = maxLost)


# ##### Initiate opencv video capture object
# 
# The `video_src` can take two values:
# 1. If `video_src=0`: OpenCV accesses the camera connected through USB
# 2. If `video_src='video_file_path'`: OpenCV will access the video file at the given path (can be MP4, AVI, etc format)

# In[8]:


video_src = "/home/tcs/john/fd1.mp4"#0
cap = cv.VideoCapture(video_src)


# ##### Start object detection and tracking

# In[9]:

fps=FPS().start()

(H, W) = (None, None)  # input image height and width for the network
writer = None
while(True):
    
    ok, image = cap.read()
    
    if not ok:
        print("Cannot read the video feed.")
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        break
    
    if W is None or H is None: (H, W) = image.shape[:2]
    
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)   # detect objects using object detection model
    
    detections_bbox = []     # bounding box for detections
    
    boxes, confidences, classIDs = [], [], []
    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > yolomodel['confidence_threshold']:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv.dnn.NMSBoxes(boxes, confidences, yolomodel["confidence_threshold"], yolomodel["threshold"])
    
    if len(idxs)>0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
            #print(type(labels[classIDs[i]]))
            if(labels[classIDs[i]]=="person"):
             detections_bbox.append((x, y, x+w, y+h))
             cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
             cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]),(x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
    
    objects = tracker.update(image,detections_bbox)           # update tracker based on the newly detected objects
    
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2) #particle predict here
        
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    cv.imshow("image", image)
    
    fps.update()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter("origoutput.avi", fourcc, 30, (W, H), True)
    writer.write(image)
writer.release()
cap.release()
cv.destroyWindow("image")

# In[8]:




