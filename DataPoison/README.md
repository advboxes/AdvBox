
Data Poison Attack desc.

mnist_paddle.py: The clean progress of train and verify with paddlepaddle framework.
poison_mnist_paddle.py: The data poison attack example with paddlepaddle. On one hand, it can be almost the same accuracy with unpoison, on the other hand, we can only poison about 5 percent of the dataset and complish a target attack, after poisoning, the model can be mislead from '3' to '4' when the trigger appears in the test picture.
poison_mnist_pytorch.py: The clean and poison code with PyTorch framework.

Usage:
Install the right version of PyTorch(>=v0.4) and PaddlePaddle(>=v1.5).
just run "Python xxx.py"
