import cv2



image2 = cv2.imread('./PILLING/IMG_0835.JPG')
gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
step = 720
k = 48
for i in range(gray_image.shape[1]/step):
    for j in range(gray_image.shape[0]/step):
        grey = gray_image[step*j:step*j+step, step*i:step*i+step]
        name = "./dataset/1/pill_"+str(k)+".jpg"
        cv2.imwrite(name,grey)
        k = k+1

