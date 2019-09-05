import sys

# choose attack method
from attack_methods.eotb_attack import EOTB_attack

# choose white-box models
from object_detectors.yolo_tiny_model_updated import YOLO_tiny_model_updated


def main(argvs):
    # choose attacker
    '''
    usage example:
        python attack_model.py \
        -fromfile data_sampling \
        -frommaskfile test/EOTB.xml \
        -fromlogofile test/logo.png (optional, you will see a logo on your sticker if you enable it)
    '''    
    attack = EOTB_attack(YOLO_tiny_model_updated)
    attack(argvs)
    
    
if __name__=='__main__':    
    main(sys.argv)
