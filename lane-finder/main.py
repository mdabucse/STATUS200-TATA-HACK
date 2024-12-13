from lane import *
from moviepy.editor import VideoFileClip


if __name__ == "__main__":

    demo = 1 

    if demo == 1:
        imagepath = 'data/test.jpg'
        img = cv2.imread(imagepath)
        img_aug = process_frame(img)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        ax2.imshow(img_aug)
        ax2.set_title('Augmented Image', fontsize=30)
        plt.show()

    else:
        video_output = 'data/project_video.mp4'
        clip1 = VideoFileClip("data/project_video.mp4")

        clip = clip1.fl_image(process_frame)
        clip.write_videofile(video_output, audio=False)

