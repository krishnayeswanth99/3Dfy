import cv2
from PIL import Image
import numpy as np
from math import ceil

def get_360(image):
    output_height = image.shape[0]
    output_width = 2 * output_height  # Twice the height for 180-degree FOV

    # Create the output image
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Mapping from output image coordinates to spherical coordinates
    for y in range(output_height):
        for x in range(output_width):
            theta = (2 * np.pi * x) / output_width  # Azimuthal angle
            phi = (np.pi * y) / output_height       # Polar angle

            # Calculate coordinates in the original image
            original_x = int((theta / (2 * np.pi)) * image.shape[1])
            original_y = int((phi / np.pi) * image.shape[0])

            # Map pixel value from original image to output image
            output_image[y, x] = image[original_y, original_x]

    return output_image

def equi_cubic_projection(image):
    h, w = image.shape[:2]
    focal_length = 0.5 * w / np.tan(0.5 * np.pi / 2)
    sphere_radius = min(w, h) / (2 * np.pi)

    equi_image = np.zeros((h, 2 * w, 3), np.uint8)

    for y in range(h):
        for x in range(w):
            theta = np.pi * (y / h - 0.5)
            phi = 2 * np.pi * (x / w - 0.5)
            x_sphere = np.cos(theta) * np.cos(phi)
            y_sphere = np.cos(theta) * np.sin(phi)
            z_sphere = np.sin(theta)
            x_proj = int((np.arctan2(y_sphere, x_sphere) / np.pi + 1) * w / 2)
            y_proj = int((np.arcsin(z_sphere) / np.pi + 0.5) * h)
            equi_image[y_proj, x_proj] = image[y, x]

    return equi_image

def make_strip(arr1, arr2, strip_size=1):
    grad = [i/strip_size for i in range(1,strip_size)]
    m = arr1.shape[1]
    print(arr1.shape, arr2.shape)

    return np.array([np.mean(arr1[:,m - strip_size + idx:,:]*(1-i), axis=1) + np.mean(arr2[:,:idx,:]*(i), axis=1) if idx>0
                     else np.mean(arr1[:,m - strip_size + idx:,:]*(1-i), axis=1)
                     for idx, i in enumerate(grad)])

def combine_images(images, strip_size=100):
    image_arr = []
    n = len(images)
    strip_cons_width = 20
    for idx in range(len(images)):
        if idx == 0:
            mid_strip = make_strip(np.flip((images[(n+idx-1)%n][:,-strip_cons_width:,:]).astype(np.float32)),
                                (images[idx][:,:strip_cons_width,:]).astype(np.float32), strip_size=strip_size)
        else:
            mid_strip = make_strip((images[(n+idx-1)%n][:,-strip_cons_width:,:]).astype(np.float32),
                                    (images[idx][:,:strip_cons_width,:]).astype(np.float32), strip_size=strip_size)
        print(np.max(mid_strip), mid_strip.shape)
        image_arr.append(np.transpose(mid_strip, (1,0,2)))
        image_arr.append(images[idx])

    return image_arr

def add_top_bottom(img, top_size=0.5, bottom_size=0.5, top_img=None, bottom_img=None, fill_top_last=False,
                   fill_bottom_last=False):
    if top_img is None and not fill_top_last:
        img_top = np.zeros((int(img.shape[0]*top_size),img.shape[1], 3), dtype=np.uint8)
    elif top_img is None and fill_top_last:
        img_top = np.zeros((int(img.shape[0]*top_size),img.shape[1], 3), dtype=np.uint8)
        for i in range(img_top.shape[0]):
            img_top[i,:,:] = img[0,:,:]
    else:
        m = top_img.shape[1]
        img_top = np.array(cv2.resize(top_img, (img.shape[1],int(img.shape[0]*top_size)), interpolation=cv2.INTER_LINEAR))

    if bottom_img is None and not fill_bottom_last:
        img_bottom = np.zeros((int(img.shape[0]*bottom_size),img.shape[1], 3), dtype=np.uint8)
    elif bottom_img is None and fill_bottom_last:
        img_bottom = np.zeros((int(img.shape[0]*bottom_size),img.shape[1], 3), dtype=np.uint8)
        for i in range(img_bottom.shape[0]):
            img_bottom[i,:,:] = img[-1,:,:]
    else:
        m = bottom_img.shape[1]
        img_bottom = np.array(cv2.resize(bottom_img, (img.shape[1],int(img.shape[0]*bottom_size)), interpolation=cv2.INTER_LINEAR))

    print(img_top.shape, img_bottom.shape)

    new_img = np.concatenate([img_top, img, img_bottom], axis=0)
    return new_img

def image_quantization(img):
    data = img.astype(np.float64) / 255.0 # normalize the data to 0 - 1
    data = 255 * data # Now scale by 255
    new_img = data.astype(np.uint8)

    return new_img