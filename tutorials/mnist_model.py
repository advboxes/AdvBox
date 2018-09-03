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

"""
CNN on mnist data using fluid api of paddlepaddle
"""
import paddle.v2 as paddle
import paddle.fluid as fluid
import os

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

def mnist_cnn_model(img):
    """
    Mnist cnn model

    Args:
        img(Varaible): the input image to be recognized

    Returns:
        Variable: the label prediction
    """
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        num_filters=20,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')#,param_attr=param_attr_1

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        num_filters=50,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')#,param_attr=param_attr_2
    fc = fluid.layers.fc(input=conv_pool_2, size=50, act='relu')#,param_attr=param_attr_3

    logits = fluid.layers.fc(input=fc, size=10, act=None)
    softmax = fluid.layers.softmax(input=logits)
    
    return softmax, logits

# transfer image to (0,1) pixel value
def _process_input(input_img, sub, div):
    res = None
    #sub, div = self._preprocess
    if np.any(sub != 0):
        res = input_img - sub
    if not np.all(sub == 1):
        if res is None:  # "res = input_ - sub" is not executed!
            res = input_img / (div) 
        else:
            res /= div
    if res is None:  # "res = (input_ - sub)/ div" is not executed!
        return input_img

    res = np.where(res==0,0.00001,res)
    res = np.where(res==1,0.99999,res) # no 0 or 1

    return res

def main(use_cuda):
    """
    Train the cnn model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    softmax, logits = mnist_cnn_model(img) # I changed the returns
    cost = fluid.layers.cross_entropy(input=softmax, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    batch_size = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=softmax, label=label, total=batch_size)

    BATCH_SIZE = 50
    PASS_NUM = 3
    ACC_THRESHOLD = 0.98
    LOSS_THRESHOLD = 10.0
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    #根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    pass_acc = fluid.average.WeightedAverage()
    for pass_id in range(PASS_NUM):
        pass_acc.reset()
        for data in train_reader():
            #print(data)
            #cw requires the image pixel between 0 and 1
            data_scaled = []
            
            for i in range(BATCH_SIZE):
                img_0_1 = _process_input(data[i][0],-1,2)
                data_scaled.append((img_0_1,data[i][1]))
                
            #print(data_scaled)
           
            loss, acc, b_size = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data_scaled),#[(img_0_1,data[0][1])]
                fetch_list=[avg_cost, batch_acc, batch_size])
            pass_acc.add(value=acc, weight=b_size)
            pass_acc_val = pass_acc.eval()[0]
            print("pass_id=" + str(pass_id) + " acc=" + str(acc[0]) +
                  " pass_acc=" + str(pass_acc_val))
            if loss < LOSS_THRESHOLD and pass_acc_val > ACC_THRESHOLD:
                # early stop
                print('early stop')
                break
        
        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc.eval()[
            0]))
    fluid.io.save_params(
        exe, dirname='./mnist', main_program=fluid.default_main_program())
    print('train mnist done')


if __name__ == '__main__':
    main(use_cuda=with_gpu)
