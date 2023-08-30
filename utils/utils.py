import os
import numpy as np
import torch
import random
import logging
import sys
from datetime import datetime

def set_device(gpu_num):
    """
    params:
        gpu_num: 0번 gpu를 쓰고 싶을 경우 "0" / 0번 1번을 같이 쓰고 싶은 경우 "0,1"
    return:
        device
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu_num
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("Current cuda device: ", torch.cuda.current_device())
    print("Count of using GPUs: ", torch.cuda.device_count())
    
    return device


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed) # numpy 연산 seed 고정
    torch.manual_seed(seed) # cpu 연산 seed 고정
    torch.cuda.manual_seed(seed) # gpu 연산 seed 고정
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def get_logger():
    logger = logging.getLogger(__name__)
    # 로그의 출력 기준 설정
    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 파일에 로그 저장 기능 추가
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # 한국 기준 년도 일 월
    date = str(datetime.now()).split(" ")[0].split("-")
    date = date[0] + date[1] + date[2]

    file_handler = logging.FileHandler(f"./logs/{date}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    return logger