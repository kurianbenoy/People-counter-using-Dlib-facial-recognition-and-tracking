from centroidtracker import CentroidTracker
from face_recognizer import FaceRecognizer
import numpy as np
import imutils
import dlib
import cv2
import urllib.request
import face_recognition

def facial_recognition():
    train_cnt=7     #number of frames to be used for creating embeddings
    fce=FaceRecognizer()
    ct = CentroidTracker(maxDisappeared=10, maxDistance=100)

    img_count=np.zeros((5),dtype=int)   #size 5 as maximum 5 people are expected to enter simultaneously.
    unknown_images= np.zeros((5,train_cnt,500,500,3),dtype=np.uint8) 
    unknown_bbox= np.zeros((5,train_cnt,4),dtype=np.uint32)

    skip_frame=7    #number of frames to be skipped between each detection
    frame_cnt=0     #counts the present frame id/no.
    face_cnt=0     #current number of faces recognized
    det_flag=False  #Gives if any faces are detected in the given frame
    i=0
    faces_per_person_added=0
    enter=False
#   cap=cv2.VideoCapture('rtsp://192.168.21.210:8080/h264_ulaw.sdp')
    cap=cv2.VideoCapture(1)

    '''Main loop '''
    while True:
        ret,frame = cap.read()
        #frame=cv2.resize(frame,(500,500))
        status = "Waiting"
        rects = []

        ''' Detection API called here '''

        if frame_cnt%skip_frame==0:
            status = "Detecting"
            trackers = []
            img, boxes, names = fce.get_faces(frame)
            if len(boxes)>=1:
                det_flag=True
            else:
                det_flag=False

            for box,name in zip(boxes,names):
                rects.append([box[0],box[1],box[2],box[3],name])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(box[0],box[1],box[2],box[3])
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

#    ''' Correlation tracker is called '''
        else:
            for (tracker, name) in zip(trackers, names):
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                rects.append([int(pos.left()), int(pos.top()),int(pos.right()), int(pos.bottom()),name])
        objects = ct.update(rects)

 #       '''Processing returned object boxes '''
        for (objectID, centroid) in objects.items():
            ''' centroid == [cX,cY,name,startX, startY, endX, endY],
                cX: Centroid X coordinate of previously tracked bbox,
                cY: Centroid X coordinate of previously tracked bbox,
                name: Person's ID or Unknown,
                Rest: Bounding box coordinates
            '''
            w=abs(int(centroid[5])-int(centroid[3]))
            h=abs(int(centroid[4])-int(centroid[6]))

            if centroid[2]=='Unknown' and frame_cnt%skip_frame==0 and det_flag==True and abs(h-w)<3:
                if img_count[objectID]<train_cnt:
                    unknown_images[objectID][img_count[objectID]]=frame
                    unknown_bbox[ objectID ][ img_count[objectID] ] = [int(centroid[4]),int(centroid[5]), int(centroid[6]),int(centroid[3])]
                    img_count[objectID]+=1

            ''' Drawing bboxes and label ids '''
            cv2.putText(frame,str(centroid[2]), (int(centroid[0]),int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 0), 2)
            cv2.rectangle(frame,(int(centroid[3]),int(centroid[4])), (int(centroid[5]),int(centroid[6])), (0, 0, 0), 2)


            if img_count[objectID]==train_cnt and enter==False:
                enter=True
                for k in range(train_cnt):
                    a=unknown_images[objectID][k]
                    '''extracting ROI for adding new faces'''
                    roi=a[ unknown_bbox[objectID][k][0]:unknown_bbox[objectID][k][2],unknown_bbox[objectID][k][3]:unknown_bbox[objectID][k][1] ]

        ''' Creating a new face embedding'''

        if img_count[i]==train_cnt:
            box=[(unknown_bbox[i][j][0],unknown_bbox[i][j][1],unknown_bbox[i][j][2],unknown_bbox[i][j][3])]
            fce.add_new_face(unknown_images[i][j],'person'+str(face_cnt),box)
            faces_per_person_added+=1
            
            
            if j==train_cnt:
                print("New person has been added to database!")
                faces_per_person_added=0
                face_cnt+=1
                img_count[i]=0
                if i==4:
                    i=0
                else:
                    i+=1
            else:
                if i==4:
                    i=0
                else:
                    i+=1

  ##############display#####################################################
        print('people count of one day=',face_cnt)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) ==ord('q'):
            break


if __name__ == "__main__":
    facial_recognition()
