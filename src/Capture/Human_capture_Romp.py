import cv2
from romp import ROMP, romp_settings

settings = romp_settings()
settings.mode = 'video'
settings.show = False
model = ROMP(settings)

bones = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),
         (8,11),(9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),
         (17,19),(18,20),(19,21)]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    outputs = model(frame)
    
    if outputs is not None and 'pj2d_org' in outputs:
        for person in outputs['pj2d_org']:
            pts = [(int(p[0]), int(p[1])) for p in person]
            
            for start, end in bones:
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
                
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.imshow('ROMP - Skeleton', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()