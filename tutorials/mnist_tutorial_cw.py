# coding: utf-8

# In[1]:

"""
CW tutorial on mnist using advbox tool.
CW method only supports targeted attack.
"""
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import paddle.fluid as fluid
import paddle.v2 as paddle

from advbox.adversary import Adversary
from advbox.attacks.CW_L2 import CW_L2
from advbox.models.paddle import PaddleModel
from tutorials.mnist_model import mnist_cnn_model


def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500
    IMG_NAME = 'img'
    LABEL_NAME = 'label'
    # create two empty program for training and variable init 
    cnn_main_program = fluid.Program()
    cnn_startup_program = fluid.Program()
    
    with fluid.program_guard(main_program=cnn_main_program, startup_program=cnn_startup_program):
        img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
        # gradient should flow
        img.stop_gradient = False
        label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
        softmax, logits = mnist_cnn_model(img)
        cost = fluid.layers.cross_entropy(input=softmax, label=label)
        avg_cost = fluid.layers.mean(x=cost)
    
    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)
    
    fluid.io.load_params(
        exe, "./mnist/", main_program=cnn_main_program)

    # advbox demo
    m = PaddleModel(
        cnn_main_program,
        cnn_startup_program,
        IMG_NAME,
        LABEL_NAME,
        
        softmax.name,
        logits.name,
        
        avg_cost.name, (-1, 1),
        channel_axis=1,
        preprocess = (-1, 2)) # x within(0,1) so we should do some transformation
    
    attack = CW_L2(m,learning_rate=0.1)
    #######
    #change parameter later
    #######
    attack_config = {"nb_classes":10,  
                     "learning_rate":0.1, 
                     "attack_iterations":50, 
                     "epsilon":0.5,
                     "targeted":True,
                     "k":0,
                     "noise":2}
    
    # use train data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in train_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])
        
        # CW_L2 targeted attack
        tlabel = 0
        adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))
            # plt.imshow(adversary.target, cmap='Greys_r')
            # plt.show()
            # np.save('adv_img', adversary.target)
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TRAIN_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break

    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # CW_L2 targeted attack
        tlabel = 0
        adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))
            # plt.imshow(adversary.target, cmap='Greys_r')
            # plt.show()
            # np.save('adv_img', adversary.target)
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("CW attack done")


if __name__ == '__main__':
    main()

