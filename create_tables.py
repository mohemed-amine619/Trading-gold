from core.settings import get_settings
from core.database.engine import init_db
settings = get_settings()
init_db(settings.database_url)
print("Database initialized.")
