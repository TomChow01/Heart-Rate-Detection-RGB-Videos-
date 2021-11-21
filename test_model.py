#Import Required Libraries
import cv2
import os
import numpy as np
import time
import sys
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from evm import *
import matplotlib.pyplot as plt
from scipy import stats
import re
from model import SampleCNN
import torch

#Heart Rate Detection Class
class HeartRate():
    def __init__(self, video_path = None, save = False,
                 color_magnify = False, mask_path = None, out_path = None):
        
        self.video_path = video_path #Path to Pre-recorded video file
        self.color_magnify = color_magnify #Whether to Color Magnify using EVM
        self.mask_path = mask_path #Path to the binary mask for subject segmentation(if available)
        self.out_path = out_path #Path to save the video file post inference
        self.save = save # (True/False)
        self.out = None
        if self.video_path:
            self.subject = video_path.replace('\\', '/').split('/')[-1].split('.')[0]
            self.text_file = open(self.subject + "_hr_new.txt", mode = "w") #Text file to record the HR
        else:
            self.text_file = open("webcam_hr.txt", mode = "w") #Text file to record the HR in case of webcam capture
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.cnn = SampleCNN().to(self.device) #Load the CNN model
        self.weight_path = 'G:\ML Projects\ISI\Baby Monitoring\Heart Rate Detection\Deploy\hr_pt_demo\cm_epoch_500.pt' #Path to pre-trained CNN weights
        self.cnn.load_state_dict(torch.load(self.weight_path)) #Load Weights
        
    def run(self):
        video = False
        
        #Load pre-recorded video file
        if self.video_path is not None and self.color_magnify is False:
            fr = 0
            cap = cv2.VideoCapture(self.video_path)
            
        #Load Webcam/IP Cam Video    
        elif self.video_path is None:
            print("Capturig video from webcam")
            ipcam_url = 'http://192.168.43.1:4747/video' # Change with own IP-Cam address
            #cap = cv2.VideoCapture(0)  #For Webcam capture
            cap = cv2.VideoCapture(ipcam_url)         
        else:
            pass
            
        fu = Face_utilities() #Face Utilities class object
        sp = Signal_processing() #Signal Processing class object
        evm = EVM() #EVM class object
        
        i=0
        #last_rects = None
        # last_shape = None
        # last_age = None
        # last_gender = None
        
        face_detect_on = False
        age_gender_on = False
    
        t = time.time()
        
        #for signal_processing
        BUFFER_SIZE = 100 # Number of previous time steps data for processing
        BPM_BUFFER_SIZE = 100 # Number of previous time steps HR (For averaging or taking the mode)
        
        
        times = []
        data_buffer = []
        bpm_buffer = []
        
        # data for plotting
        filtered_data = []
        
        fft_of_interest = []
        freqs_of_interest = []
        
        bpm = 0
        
        # Color Magnification
        if self.color_magnify:
            video_tensor = evm.magnify_color(video_name = self.video_path, mask = self.mask_path, out_name = self.out_path,
                     low = 1, high = 1.7, save = True, skin_detect = False, inpaint_bg = False, amplification= 10)
            cap = cv2.VideoCapture(self.out_path)
            
        
        while True:
            # grab a frame -> face detection -> crop the face -> 68 facial landmarks -> get mask from those landmarks
    
            # calculate time for each loop
            t0 = time.time()
            
            if(i%1==0):
                face_detect_on = True
                if(i%10==0):
                    age_gender_on = True
                else:
                    age_gender_on = False
            else: 
                face_detect_on = False
            
            # For Pre-recorded video
            if self.video_path:
                ret, frame = cap.read() # Take a particular frame from the video
            
            # For webcam capture
            elif self.video_path is None:
                ret, frame = cap.read() # Take a particular frame from the video
                
            else:
                frame = video_tensor[fr]
                fr += 1

            
            if frame is None:
                print("End of video")
                cv2.destroyAllWindows()
                break
                        
            
            # Frame with detected face
            ret_process = fu.no_age_gender_face_process(frame, "68")
            
            # No face detected
            if ret_process is None:
                cv2.putText(frame, "No face detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
                cv2.imshow("frame",frame)
                print(time.time()-t0)
                
                cv2.destroyWindow("face")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                continue
            
            rects, face, shape, aligned_face, aligned_shape = ret_process
            
            (x, y, w, h) = face_utils.rect_to_bb(rects) # coordinates of bounding box
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # Draw rectangle
            
            
            if(len(aligned_shape)==68):
                cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                        (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
                cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                        (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
            else:
                cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                            (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
                
                cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                            (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
            
            for (x, y) in aligned_shape: 
                cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
                
                
            #for signal_processing
            ROIs = fu.ROI_extraction(aligned_face, aligned_shape)
            # Extract green color channel value
            green_val = sp.extract_color(ROIs)
            
            data_buffer.append(green_val)
            
            if(video==False):
                times.append(time.time() - t)
            else:
                times.append((1.0/298.)*i)
                pass
            
            L = len(data_buffer)
            c = 0
            
            if L > BUFFER_SIZE:
                data_buffer = data_buffer[-BUFFER_SIZE:]
                times = times[-BUFFER_SIZE:]
                #bpms = bpms[-BUFFER_SIZE//2:]
                L = BUFFER_SIZE
            #print(times)
            if L==100:
                c += 1 
                fps = float(L) / (times[-1] - times[0])
                cv2.putText(frame, "fps: {0:.2f}".format(fps), (int(frame.shape[0]*0.05), 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                #
                detrended_data = sp.signal_detrending(data_buffer)
                
                # Signal interpolation
                try:
                    interpolated_data = sp.interpolation(detrended_data, times)
                except:
                    interpolated_data = detrended_data
                    
                
                smoothed_data = sp.median_filter(interpolated_data, 15) #Smoothing using median filter
                # rm = sp.running_mean(interpolated_data, 3)

                
                normalized_data = sp.normalization(interpolated_data)
                normalized_data_med = sp.normalization(smoothed_data) #Normalization
                #normalized_data_rm = sp.normalization(rm)
                
                # Apply FFT
                fft_of_interest, freqs_of_interest = sp.fft(normalized_data, fps)
                fft_of_interest_med, freqs_of_interest_med = sp.fft(normalized_data_med, fps)

                
                #Preparing the data for Model
                x = np.zeros((1, 36))

                x[:, :len(fft_of_interest_med)] = np.array(fft_of_interest_med)
                x = torch.tensor(x).unsqueeze(0).to(self.device)
                
                y = self.cnn(x.to(torch.float32)) #Inference
                bpm_med = int(y.item())

                
                
                
                #Final bpm using statistical mode
                bpm_buffer.append(bpm_med)
                if len(bpm_buffer) > BPM_BUFFER_SIZE:
                    bpm_med = stats.mode(bpm_buffer[-BPM_BUFFER_SIZE:])[0][0]
                    
                
                
                
                # Write the HR on frame
                #cv2.putText(frame, "HR: {0:.2f}".format(bpm), (30,int(frame.shape[0]*0.25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, "HR: {0:.2f}".format(bpm_med), (int(frame.shape[0]*0.05), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Normal/ Higher/ Lower Range
                if bpm_med < 60:
                    cv2.putText(frame, "HR is Lower than Normal", (int(frame.shape[0]*0.05), int(frame.shape[0]*0.8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                elif bpm_med > 85:
                    cv2.putText(frame, "HR is Higher than Normal", (int(frame.shape[0]*0.05), int(frame.shape[0]*0.8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
                else:
                    cv2.putText(frame, "HR is in Normal range", (int(frame.shape[0]*0.05), int(frame.shape[0]*0.8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                

                #Write to text file
                self.text_file.write("time: {0:.4f} ".format(times[-1]) + ", HR: {0:.2f} ".format(bpm_med) + "\n")
                
                #Pass the final frame to display on web browser
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_en = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_en + b'\r\n')            

            
            
            #save
            if self.save:
                self.out = cv2.VideoWriter(self.out_path, -1, 20.0, (frame.shape[1],frame.shape[0]))
            if self.save and frame is not None:
                print("Saving Video...")
                self.out.write(frame)

            i = i+1
            
            # waitKey to show the frame and break loop whenever 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
        # Destroy window after processing
        cap.release()
        if self.save:
            self.out.release()
        cv2.destroyAllWindows()
        self.text_file.close()
            
        
    

    

if __name__ == "__main__":
    vid_dir = 'G:/ML Projects/ISI/Baby Monitoring/stable videos/ISI/normal_light'
    video_path = os.path.join(vid_dir, 'sub_1.mp4')
    
    out_dir = "G:/ML Projects/ISI/Motion Microscopy/Heart Rate Detection/Outputs/deploy"
    out_path = os.path.join(out_dir, 'sub_1.mp4')
    hr = HeartRate(video_path = video_path, save = True, color_magnify = False,
                           mask_path = None, out_path = out_path)
    hr.run()
            
            
