import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 1080p kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

alpha = 0.03

smooth_target = None
smooth_radius = None
smooth_laser = None

zoom = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # dijital zoom
    frame = cv2.resize(frame, None, fx=zoom, fy=zoom)

    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # --- DAİRE TESPİTİ (uzak mesafe ayarlı) ---
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=100,
        param2=20,
        minRadius=3,
        maxRadius=120
    )

    target = None

    if circles is not None:
        circles = np.uint16(np.around(circles))

        if smooth_target is None:
            c = max(circles[0], key=lambda x: x[2])
        else:
            cx, cy = smooth_target
            c = min(
                circles[0],
                key=lambda x: np.sqrt((x[0]-cx)**2 + (x[1]-cy)**2)
            )

        target = (c[0], c[1], c[2])


    if target:

        if smooth_target is None:
            smooth_target = np.array(target[:2], dtype=float)
            smooth_radius = target[2]

        else:
            smooth_target = (1-alpha)*smooth_target + alpha*np.array(target[:2])
            smooth_radius = (1-alpha)*smooth_radius + alpha*target[2]

        cx, cy = smooth_target.astype(int)
        r = int(smooth_radius)

        # stabil YEŞİL DAİRE
        cv2.circle(frame, (cx,cy), r, (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

        # sarı hedef kilidi
        cv2.circle(frame, (cx,cy), 8, (0,255,255), 2)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,120,200])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,200])
    upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    laser = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x,y),r = cv2.minEnclosingCircle(c)

        if r > 2:
            laser = (int(x),int(y))

            if smooth_laser is None:
                smooth_laser = np.array(laser,dtype=float)
            else:
                smooth_laser = (1-alpha)*smooth_laser + alpha*np.array(laser)

            lx,ly = smooth_laser.astype(int)

            cv2.circle(frame,(lx,ly),6,(0,0,255),-1)


    h,w,_ = frame.shape
    center = (w//2, h//2)

    cv2.line(frame,(center[0]-40,center[1]),(center[0]+40,center[1]),(255,255,255),1)
    cv2.line(frame,(center[0],center[1]-40),(center[0],center[1]+40),(255,255,255),1)

    
    if smooth_laser is not None and smooth_target is not None:

        lx,ly = smooth_laser.astype(int)
        tx,ty = smooth_target.astype(int)

        cv2.line(frame,(lx,ly),(tx,ty),(255,255,0),2)

    cv2.imshow("SNIPER TRACKING SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()