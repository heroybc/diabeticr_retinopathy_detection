
# 将图像中圆环区域展开成矩形长条的方法Python实现
def get_huan_by_circle(img,circle_center,radius,radius_width):
    black_img = np.zeros((radius_width,int(2*radius*math.pi),3),dtype='uint8')
    for row in range(0,black_img.shape[0]):
        for col in range(0,black_img.shape[1]):
            theta = math.pi*2/black_img.shape[1]*(col+1)
            rho = radius-row-1
            p_x = int(circle_center[0] + rho*math.cos(theta)+0.5)
            p_y = int(circle_center[1] - rho*math.sin(theta)+0.5)
            
            black_img[row,col,:] = img[p_y,p_x,:]
            
    IM.fromarray(black_img).show()
    return black_img