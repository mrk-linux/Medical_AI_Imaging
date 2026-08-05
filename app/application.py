from config import Config


class Application:
    """Main application controller."""

    def __init__(self) -> None:
        self.config = Config()

    def run(self) -> None:
        print(f"{self.config.APP_NAME} v{self.config.VERSION}")