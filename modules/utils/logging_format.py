import logging


class CustomFormatter(logging.Formatter):
    """CustomFormatter class
    logging Formatter를 상속받아 customize한 클래스
    상태에 따라 log를 기록할 방식 설정(글자색, 형식)

    Attributes:
        __색상 (str): 색상으로 출력하기 위한 ANSI escape 코드를 저장하는 문자열 변수
        __reset (str): 이전에 설정한 모든 ANSI escape 코드를 초기화 하는 문자열 변수
        __log_format (str): 로그 형식 문자열을 저장하는 변수
        __FORMATS (dict): 로그 레벨에 따라 로그 포맷을 지정
    """
    __grey = "\x1b[38;21m"
    __yellow = "\x1b[33;21m"
    __green = "\x1b[32m"
    __red = "\x1b[31;21m"
    __bold_red = "\x1b[31;1m"
    __blue = "\x1b[1;34m"
    __light_blue = "\x1b[1;36m"
    __purple = "\x1b[1;35m"
    __reset = "\x1b[0m"

    __log_format = (
        "[{}%(levelname)s:%(asctime)s %(name)s{}] %(message)s (%(filename)s:%(lineno)d)"
    )

    __FORMATS = {
        logging.DEBUG: __log_format.format(__light_blue, __reset),
        logging.INFO: __log_format.format(__green, __reset),
        logging.WARNING: __log_format.format(__yellow, __reset),
        logging.ERROR: __log_format.format(__red, __reset),
        logging.CRITICAL: __log_format.format(__bold_red, __reset),
    }

    def format(self, record) -> str:
        """log format에 따라 log 기록

        Args:
            record (str): log에 기록할 record

        Returns:
            str : 설정한 format에 맞춘 record
        """
        log_fmt = self.__FORMATS.get(record.levelno)

        formatter = logging.Formatter(
            log_fmt,
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        return formatter.format(record)
