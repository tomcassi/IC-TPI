import imageio as iio
img = iio.imread("img.png")
img[0][0]= [255,0,0,255]

iio.imwrite("g4g.png", img)