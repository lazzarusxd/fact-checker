import threading
import webbrowser

import uvicorn

from app.config import settings


def _open_browser() -> None:
    webbrowser.open(f"http://{settings.host}:{settings.port}")


if __name__ == "__main__":
    threading.Timer(3.0, _open_browser).start()
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False)
