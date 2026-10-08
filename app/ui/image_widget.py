from PySide6.QtWidgets import (
    QLabel,
    QFrame,
    QVBoxLayout,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ImageWidget(QFrame):
    """Widget for displaying X-Ray images."""

    def __init__(self) -> None:
        super().__init__()

        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        self.image_label = QLabel()

        self.image_label.setAlignment(
            Qt.AlignCenter
        )

        self.image_label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                border-radius: 15px;
            }
            """
        )

        layout.addWidget(
            self.image_label
        )

        self.setLayout(
            layout
        )

        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border-radius: 15px;
            }
            """
        )

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()

        shadow.setBlurRadius(25)
        shadow.setOffset(0, 8)

        self.setGraphicsEffect(
            shadow
        )

        self.current_pixmap = None

    def load_image(
        self,
        image_path: str,
    ) -> None:
        """Load and display image."""

        pixmap = QPixmap(
            image_path
        )

        if pixmap.isNull():
            self.image_label.setText(
                "Unable to load image"
            )
            return

        self.current_pixmap = pixmap

        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
        )

        self.image_label.setPixmap(
            scaled_pixmap
        )