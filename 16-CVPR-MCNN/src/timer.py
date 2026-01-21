import time

class Timer(object):
    """
    훈련 또는 테스트 중 이미지 한 장/배치 한 개를 처리하는 데 걸리는 
    시간을 측정하고 평균값을 계산해주는 클래스입니다.
    """
    def __init__(self):
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        """시간 측정 시작 (Start clock)"""
        # time.clock은 Python 3.8 이상에서 삭제되었으므로
        # 호환성을 위해 time.time() 또는 time.perf_counter()를 사용합니다.
        self.start_time = time.time()

    def toc(self, average=True):
        """시간 측정 종료 및 기록 (Stop clock)"""
        self.diff = time.time() - self.start_time
        self.tot_time += self.diff
        self.calls += 1
        self.average_time = self.tot_time / self.calls
        
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        """측정값 초기화"""
        self.tot_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.