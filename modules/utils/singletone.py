class Singletone:
    """Singletone Class
        싱글톤 디자인 패턴을 구현하는 클래스
        
    Attributes:
        __instance: 클래스의 인스턴스 저장
        __getInstances: __instance 값을 반환하는 클래스 메소드
        instance: 인스턴스가 없으면 새로운 인스턴스를 생성하고 있으면 기존 인스턴스를 반환, 이후 __getInstance에 덮어씀
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
