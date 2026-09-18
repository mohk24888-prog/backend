import os
import sys
import types

os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./footiq_minimal.db'
os.environ['database_url_sync'] = 'sqlite:///./footiq_minimal.db'

torch_mod = types.ModuleType('torch')
class _Device:
    type = 'cpu'
torch_mod.device = _Device
torch_mod.cuda = types.ModuleType('cuda')
torch_mod.cuda.is_available = lambda: False
sys.modules['torch'] = torch_mod

ultralytics_mod = types.ModuleType('ultralytics')
class _YOLO: pass
ultralytics_mod.YOLO = _YOLO
sys.modules['ultralytics'] = ultralytics_mod

sys.path.insert(0, '.')

import uvicorn
from api.main_minimal import app

uvicorn.run(app, host='127.0.0.1', port=8001, log_level='info')
