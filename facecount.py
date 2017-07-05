import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

cap.set(3, 440);
cap.set(4, 320);




while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    s_img = cv2.imread('thuglife.png', -1)

    c, d, f = img.shape
    # result = np.zeros((c,d,3),np.uint8)

            #result = np.zeros((a,b,3),np.uint8)

    

    facecount = 1
    for (x,y,w,h) in faces:
        cv2.putText(img,"face " + str(facecount), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        facecount = facecount + 1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)


        ix = 0
        iy = 0
        iw = 0
        ih = 0 

        ix1 = 0 
        iy1 = 0
        iw1 = 0
        ih1 = 0
        
        count = 0
        for (ex,ey,ew,eh) in eyes:
        	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        	if count == 1 :
        		ix = ex
        		iy = ey
        		iw = ew
        		ih = eh
        	else :
        		ix1 = ex
        		iy1 = ey
        		iw1 = ew
        		ih1 = eh
        	count = count + 1


          #  cv2.imshow('dst_rt', img)


 		# Re-calculate the width and height of the mustache image
        gwidth = (ix+iw) - ix1
        gheight = ih
        
        nx = 0 
        ny = 0
        nw = 0
        nh = 0

        if ix < ix1 :
        	gwidth = (ix1+iw1) - ix
        	nx = ix
        	ny = iy
        	nw = gwidth
        	nh = gheight
        if ix > ix1 :
        	width = (ix+iw) - ix1
        	nx = ix1 
        	ny = iy1
        	nw = gwidth
        	nh = gheight
		
	# Center the mustache on the bottom of the nose
        x1 = nx - (gwidth/4)
        x2 = nx + nw + (gwidth/4)
        y1 = ny + nh - (gheight/2)
        y2 = ny + nh + (gheight/2)

		# Check for clipping
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h
	# Re-calculate the width and height of the mustache image
        gwidth = x2 - x1
        gheight = y2 - y1
        # img[273:333, 100:160] = s_img
        try : 
            x_offset= y_offset=60
            r = w / s_img.shape[1]
            dim = (w, int(s_img.shape[0] * r))
            resized = cv2.resize(s_img, dim, interpolation = cv2.INTER_AREA)
            #img[y+y_offset:y+y_offset+resized.shape[0], x:x+resized.shape[1]] = resized

            a, b,  depth =  resized.shape

            result = np.zeros((a,b,3),np.uint8)

            for i in range(a):
                for j in range(b):
                    color1=img[i+y+y_offset,j+x]
                    try :
                        color2=resized[i,j]
                        alpha = color2[3] /255.0
                        new_color = [ (1 - alpha) * color1[0] + alpha * color2[0], (1 - alpha) * color1[1] + alpha * color2[1], (1 - alpha) * color1[2] + alpha * color2[2] ]
                        result[i, j] = new_color
                    except:
                        result[i,j] = color1

            img[y+y_offset:y+y_offset+resized.shape[0], x:x+resized.shape[1]] = result

            # for i in range(c):
            #     for j in range(d):
            #         color1=img[i,j]
            #         if i >= (y+y_offset)  and j >= x :
            #             try :
            #             # if i >= y+y_offset and j >= x :
            #                 color2=resized[i-y+y_offset, j-x]
            #                 alpha = color2[3] /255.0
            #                 new_color = [ (1 - alpha) * color1[0] + alpha * color2[0], (1 - alpha) * color1[1] + alpha * color2[1], (1 - alpha) * color1[2] + alpha * color2[2] ]
            #                 result[i, j] = new_color
            #         # else :
            #         #     result[i,j] = color1
            #             except:
            #                 result[i,j] = color1
            #         else: 
            #             result[i,j] = color1

        # a,b,c = resized.shape

        # magenta = np.array([255, 0, 255], dtype=np.uint8)

        # overlayWithTransparencyBoring(img, resized, y+y_offset, x, magenta)

        # gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # eyes_cascade = cv2.CascadeClassifier('frontaleyes.xml')
        # eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)

        # for (d,e,f,g) in eyes:
        #     cv2.rectangle(gray,(d,e), (d,e,f,g),(255,86,30), 3)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # for i in range(0,a):
        #     for j in range (0, b):
        #         if resized[i,j][3] != 0:
        #             img[y+i, x+j] = resized[i,j]

        # img = a
        # for c in range(0,3):
        #     # img[y+y_offset:y+y_offset+resized.shape[0], x:x+resized.shape[1]] = resized[:,:,c] * 
        #     # (resized[:,:,3]/255.0) +  l_img[y+y_offset:y+y_offset+resized.shape[0], x_offset:x+resized.shape[1], c] * (1.0 - resized[:,:,3]/255.0)
        #     alpha = resized[:, :, 3] / 255.0
        #     color = resized[:, :, c] * (1.0-alpha)
        #     beta  = img[y+y_offset:y+y_offset+resized.shape[0], x_offset:x+resized.shape[1], c] * (alpha)
        #     img[y+y_offset:y+y_offset+resized.shape[0], x:x+resized.shape[1]] = color + beta
        except :
            break
        # s_img[:,:,c] * (s_img[:,:,3]/255.0) +  l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)


    		


    # cv2.imshow('img',img)        
    cv2.imshow('glass', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
