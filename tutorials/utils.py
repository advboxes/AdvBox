#coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt


#测试代码
def show():
    print("测试代码")

#对比展现原始图片和对抗样本图片
def show_images_diff(original_img,adversarial_img):

    plt.figure()

    original_img=original_img/255.0
    adversarial_img=adversarial_img/255.0

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial Image')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = adversarial_img - original_img
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max()/2.0+0.5
    print(difference)
    plt.imshow(difference)
    plt.axis('off')

    plt.show()

#自定义的视频图像处理函数
def invert_green_blue(image):
    return image[:,:,[0,2,1]]

#处理视频文件
def do_movie_content(infile,outfile):
    # pip install moviepy
    #pip install requests
    #brew install imagemagick 如果更新慢
    # cd "$(brew --repo)" && git remote set-url origin https://git.coding.net/homebrew/homebrew.git
    # Import everything needed to edit video clips
    ## From tensorflow/models/research/
    #export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    from moviepy.editor import *

    # Load myHolidays.mp4 and select the subclip 00:00:50 - 00:00:60
    clip = VideoFileClip(infile).subclip(0, 10)

    modifiedClip = clip.fl_image(invert_green_blue)


    #添加字幕
    # Generate a text clip. You can customize the font, color, etc.
    txt_clip = TextClip("Dou Goodman", fontsize=70, color='white')

    # Say that you want it to appear 10s at the center of the screen
    txt_clip = txt_clip.set_pos('center').set_duration(10)

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([modifiedClip, txt_clip])

    # Write the result to a file (many options available !)
    video.write_videofile(outfile)


if __name__ == '__main__':
    do_movie_content('baidu.mp4','baidu-edited.mp4')