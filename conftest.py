"""Put the project root on sys.path so tests can `import clean_data`.

pytest prepends the directory of each conftest.py it finds, so simply having
this file at the root is what makes the import work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
