class Singletone:
    """Singletone Class
        싱글톤 디자인 패턴을 구현하는 클래스
        
    """
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance
