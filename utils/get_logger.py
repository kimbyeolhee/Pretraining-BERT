import logging
import os
import sys
from datetime import datetime


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