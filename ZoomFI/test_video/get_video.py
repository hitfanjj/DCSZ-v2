import cv2
import os
import numpy as np
import glob

def crop_center(hr, ph, pw):
    ih, iw = hr.shape[0:2]
    lr_patch_h, lr_patch_w = ph, pw
    ph = ih // 2 - lr_patch_h // 2
    pw = iw // 2 - lr_patch_w // 2
    return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

def merge_image_to_video(folder_name, id, out_path):
    fps = 5
    firstflag = True
    files1 = sorted(glob.glob(os.path.join(folder_name, '*.png')))
    for i in range(len(files1)):
        filename = files1[i]
        frame = cv2.imread(filename)
        if firstflag == True:  # 读取第一张图时进行初始化，尺寸也近照些图
            firstflag = False
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            img_size = (frame.shape[1], frame.shape[0])
            video = cv2.VideoWriter(os.path.join(out_path, id+".mp4"), fourcc, fps, img_size)
        else:
            frame_suitable = cv2.resize(frame, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
            video.write(frame_suitable)
    video.release()


if __name__ == '__main__':

    from_path = "test images path"
    out_path = "output videos path"

    ids = sorted(os.listdir(from_path))
    print(ids)
    for id in ids:
        folder_name = os.path.join(from_path, id)
        if os.path.isfile(folder_name) == False:
            merge_image_to_video(folder_name, id, out_path)