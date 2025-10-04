from robocore.utils.path import get_robocore_path
from robocore.utils.beauty_logger import beauty_print

beauty_print(get_robocore_path())
beauty_print(get_robocore_path("assets/robot/urdf"))
beauty_print(get_robocore_path("assets/robot/urdf/122"))  # should raise FileNotFoundError
