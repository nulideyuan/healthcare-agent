import os
import tempfile

# Tests must never write to the real conversation-memory database.
os.environ["HC_MEMORY_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="hc_test_"), "memory.sqlite")
