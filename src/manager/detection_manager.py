
from src.utils.logger import Logger
from src.model.scrfd.scrfd import SCRFD
from src.model.scrfd.scrfd_trt import SCRFDTRT



class DetectionManager:
    def __init__(self, config, engine_name) -> None:
        self.logger = Logger().get_logger()
        self.engine = None
        if engine_name == "scrfd_onnx":
            self.engine = SCRFD(config)
        elif engine_name == "scrfd_trt":
            self.engine = SCRFDTRT(config)
        else:
            self.logger.error(f"Does not suppot {engine_name} engine name")
            exit()
            
    def get_engine(self):
        return self.engine