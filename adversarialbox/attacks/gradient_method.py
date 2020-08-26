"""
This module provide the attack method for Iterator FGSM's implement.
"""
from __future__ import division

from builtins import range
import logging
from collections import Iterable

import numpy as np


from .base import Attack
from functools import reduce

__all__ = [
    'GradientMethodAttack', 'FastGradientSignMethodAttack', 'FGSM',
    'FastGradientSignMethodTargetedAttack', 'FGSMT',
    'BasicIterativeMethodAttack', 'BIM',
    'IterativeLeastLikelyClassMethodAttack', 'ILCM', 'MomentumIteratorAttack',
    'MIFGSM','FGSM_static'
]




class GradientMethodAttack(Attack):
    """
    This class implements gradient attack method, and is the base of FGSM, BIM,
    ILCM, etc.
    """

    def __init__(self, model, support_targeted=True):
        """
        :param model(model): The model to be attacked.
        :param support_targeted(bool): Does this attack method support targeted.
        """
        super(GradientMethodAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               norm_ord=np.inf,
               epsilons=0.01,
               epsilons_max=0.5,
               steps=10,
               epsilon_steps=100):
        #epsilons_max 为动态调整epsilon时的上限 静态epsilon算法时epsilon_steps=1  epsilons_max=epsilons即可
        """
        Apply the gradient attack method.
        :param adversary(Adversary):
            The Adversary object.
        :param norm_ord(int):
            Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
        :param epsilons(list|tuple|int):
            Attack step size (input variation).
            Largest step size if epsilons is not iterable.
        :param steps:
            The number of attack iteration.
        :param epsilon_steps:
            The number of Epsilons' iteration for each attack iteration.
        :return:
            adversary(Adversary): The Adversary object.
        """
        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        logging.info('epsilons={0},epsilons_max={1},steps={2},epsilon_steps={3}'.
                     format(epsilons,epsilons_max,steps,epsilon_steps))

        if not isinstance(epsilons, Iterable):
            #从epsilons到0.5逐步增大
            epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()


        #assert self.model.channel_axis() == adversary.original.ndim
        assert (self.model.channel_axis() == 1 or
                self.model.channel_axis() == adversary.original.shape[0] or
                self.model.channel_axis() == adversary.original.shape[-1])



        #从[epsilon,0.5]动态调整epsilon 直到攻击成功
        for epsilon in epsilons[:]:
            step = 1
            #强制拷贝 避免针对adv_img的修改也影响adversary.original
            adv_img = np.copy(adversary.original)
            if epsilon == 0.0:
                continue
            for i in range(steps):
                if adversary.is_targeted_attack:
                    gradient = +self.model.gradient(adv_img,
                                                    adversary.target_label)
                else:
                    gradient = -self.model.gradient(adv_img,
                                                   adversary.original_label)
                if norm_ord == np.inf:
                    gradient_norm = np.sign(gradient)
                else:
                    gradient_norm = gradient / self._norm(
                        gradient, ord=norm_ord)

                #logging.info('epsilon * gradient_norm={0}'.format(gradient_norm * epsilon))
                #logging.info('epsilon * gradient_norm* (max_ - min_)={0}'.format(gradient_norm * epsilon* (max_ - min_)))
                #改进的实现 不用考虑特征取值范围
                #adv_img = adv_img + epsilon * gradient_norm * (max_ - min_)
                #按照论文实现
                adv_img = adv_img + epsilon * gradient_norm

                adv_img = np.clip(adv_img, min_, max_)
                adv_label = np.argmax(self.model.predict(adv_img))
                logging.info('step={}, epsilon = {:.5f}, pre_label = {}, adv_label={} logits={}'.
                             format(step, epsilon, pre_label,adv_label,self.model.predict(adv_img)[adv_label]))
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary
                step += 1
        return adversary

    @staticmethod
    def _norm(a, ord):
        if a.ndim == 1:
            return np.linalg.norm(a, ord=ord)
        if a.ndim == a.shape[0]:
            norm_shape = (a.ndim, reduce(np.dot, a.shape[1:]))
            norm_axis = 1
        else:
            norm_shape = (reduce(np.dot, a.shape[:-1]), a.ndim)
            norm_axis = 0
        return np.linalg.norm(a.reshape(norm_shape), ord=ord, axis=norm_axis)


class FastGradientSignMethodTargetedAttack(GradientMethodAttack):
    """
    "Fast Gradient Sign Method" is extended to support targeted attack.
    "Fast Gradient Sign Method" was originally implemented by Goodfellow et
    al. (2015) with the infinity norm.

    Paper link: https://arxiv.org/abs/1412.6572
    """

    #硬编码了epsilons=0.01
    def _apply(self, adversary, epsilons=0.01,epsilons_max=0.5,epsilon_steps=100,steps=10):
        return GradientMethodAttack._apply(
            self,
            adversary=adversary,
            norm_ord=np.inf,
            epsilons=epsilons,
            epsilons_max=epsilons_max,
            steps=steps,
            epsilon_steps=epsilon_steps)



class FastGradientSignMethodAttack(FastGradientSignMethodTargetedAttack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm, and is known as the "Fast Gradient Sign Method".

    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model):
        super(FastGradientSignMethodAttack, self).__init__(model, False)


class IterativeLeastLikelyClassMethodAttack(GradientMethodAttack):
    """
    "Iterative Least-likely Class Method (ILCM)" extends "BIM" to support
    targeted attack.
    "The Basic Iterative Method (BIM)" is to extend "FSGM". "BIM" iteratively
    take multiple small steps while adjusting the direction after each step.

    Paper link: https://arxiv.org/abs/1607.02533
    """

    def _apply(self, adversary, epsilons=0.01, epsilons_max=0.5,steps=1000,epsilon_steps=1000):
        return GradientMethodAttack._apply(
            self,
            adversary=adversary,
            norm_ord=np.inf,
            epsilons=epsilons,
            epsilons_max=epsilons_max,
            steps=steps,
            epsilon_steps=epsilon_steps)


class BasicIterativeMethodAttack(IterativeLeastLikelyClassMethodAttack):
    """
    FGSM is a one-step method. "The Basic Iterative Method (BIM)" iteratively
    take multiple small steps while adjusting the direction after each step.
    Paper link: https://arxiv.org/abs/1607.02533
    """

    def __init__(self, model):
        super(BasicIterativeMethodAttack, self).__init__(model, False)


class MomentumIteratorAttack(GradientMethodAttack):
    """
    The Momentum Iterative Fast Gradient Sign Method (Dong et al. 2017).
    This method won the first places in NIPS 2017 Non-targeted Adversarial
    Attacks and Targeted Adversarial Attacks. The original paper used
    hard labels for this attack; no label smoothing. inf norm.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, support_targeted=True):
        """
        :param model(model): The model to be attacked.
        :param support_targeted(bool): Does this attack method support targeted.
        """
        super(MomentumIteratorAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               norm_ord=np.inf,
               epsilons=0.1,
               epsilons_max=0.5,
               steps=100,
               epsilon_steps=100,
               decay_factor=1):
        """
        Apply the momentum iterative gradient attack method.
        :param adversary(Adversary):
            The Adversary object.
        :param norm_ord(int):
            Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
        :param epsilons(list|tuple|float):
            Attack step size (input variation).
            Largest step size if epsilons is not iterable.
        :param epsilon_steps:
            The number of Epsilons' iteration for each attack iteration.
        :param steps:
            The number of attack iteration.
        :param decay_factor:
            The decay factor for the momentum term.
        :return:
            adversary(Adversary): The Adversary object.
        """
        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        assert self.model.channel_axis() == adversary.original.ndim
        assert (self.model.channel_axis() == 1 or
                self.model.channel_axis() == adversary.original.shape[0] or
                self.model.channel_axis() == adversary.original.shape[-1])

        if not isinstance(epsilons, Iterable):
            #epsilons = np.linspace(0, epsilons, num=epsilon_steps)
            #从epsilons到epsilons_max逐步增大
            epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

        min_, max_ = self.model.bounds()
        pre_label = adversary.original_label

        for epsilon in epsilons[:]:
            if epsilon == 0.0:
                continue
            step = 1
            adv_img = adversary.original
            momentum = 0
            for i in range(steps):
                if adversary.is_targeted_attack:
                    gradient = -self.model.gradient(adv_img,
                                                    adversary.target_label)
                else:
                    gradient = self.model.gradient(adv_img, pre_label)

                # normalize gradient
                velocity = gradient / self._norm(gradient, ord=1)
                momentum = decay_factor * momentum + velocity
                if norm_ord == np.inf:
                    normalized_grad = np.sign(momentum)
                else:
                    normalized_grad = self._norm(momentum, ord=norm_ord)
                perturbation = epsilon * normalized_grad
                adv_img = adv_img + perturbation
                adv_img = np.clip(adv_img, min_, max_)
                adv_label = np.argmax(self.model.predict(adv_img))
                logging.info(
                    'step={}, epsilon = {:.5f}, pre_label = {}, adv_label={}'
                    .format(step, epsilon, pre_label, adv_label))
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary
                step += 1

        return adversary




class FGSMSAttack(Attack):
    """
    静态FGSM epsilon静态
    """

    def __init__(self, model, support_targeted=True):
        """
        :param model(model): The model to be attacked.
        :param support_targeted(bool): Does this attack method support targeted.
        """
        super(FGSMSAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,
               adversary,
               norm_ord=np.inf,
               epsilon=0.01,
               steps=10):
        """
        Apply the gradient attack method.
        :param adversary(Adversary):
            The Adversary object.
        :param norm_ord(int):
            Order of the norm, such as np.inf, 1, 2, etc. It can't be 0.
        :param epsilons(list|tuple|int):
            Attack step size (input variation).
            Largest step size if epsilons is not iterable.
        :param steps:
            The number of attack iteration.
        :param epsilon_steps:
            The number of Epsilons' iteration for each attack iteration.
        :return:
            adversary(Adversary): The Adversary object.
        """
        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        #assert self.model.channel_axis() == adversary.original.ndim
        assert (self.model.channel_axis() == 1 or
                self.model.channel_axis() == adversary.original.shape[0] or
                self.model.channel_axis() == adversary.original.shape[-1])

        step = 1
        #强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.original)
        for i in range(steps):
            if adversary.is_targeted_attack:
                gradient = -self.model.gradient(adv_img,
                                                adversary.target_label)
            else:
                gradient = self.model.gradient(adv_img,
                                               adversary.original_label)
            if norm_ord == np.inf:
                gradient_norm = np.sign(gradient)
            else:
                gradient_norm = gradient / self._norm(
                    gradient, ord=norm_ord)

            adv_img = adv_img + epsilon * gradient_norm * (max_ - min_)
            adv_img = np.clip(adv_img, min_, max_)
            adv_label = np.argmax(self.model.predict(adv_img))
            logging.info('step={}, epsilon = {:.5f}, pre_label = {}, '
                         'adv_label={}'.format(step, epsilon, pre_label,
                                               adv_label))
            if adversary.try_accept_the_example(adv_img, adv_label):
                return adversary
            step += 1


        return adversary

    @staticmethod
    def _norm(a, ord):
        if a.ndim == 1:
            return np.linalg.norm(a, ord=ord)
        if a.ndim == a.shape[0]:
            norm_shape = (a.ndim, reduce(np.dot, a.shape[1:]))
            norm_axis = 1
        else:
            norm_shape = (reduce(np.dot, a.shape[:-1]), a.ndim)
            norm_axis = 0
        return np.linalg.norm(a.reshape(norm_shape), ord=ord, axis=norm_axis)


FGSM = FastGradientSignMethodAttack
FGSMT = FastGradientSignMethodTargetedAttack
BIM = BasicIterativeMethodAttack
ILCM = IterativeLeastLikelyClassMethodAttack
MIFGSM = MomentumIteratorAttack
FGSM_static = FGSMSAttack
