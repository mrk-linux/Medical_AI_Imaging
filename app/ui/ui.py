import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
)
from PySide6.QtCore import Qt

from app.ui.image_widget import ImageWidget
from app.ui.result_widget import ResultWidget
from app.ui.inference_controller import InferenceController
from app.ui.styles import APP_STYLE


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.selected_image_path = None

        self.inference_controller = InferenceController()

        self.setWindowTitle("Medical AI Imaging")
        self.resize(1200, 800)

        self.create_ui()

    def create_ui(self) -> None:
        """Create main user interface."""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()

        title = QLabel(
            "Medical AI Imaging - Chest X-Ray Analysis"
        )
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)

        content_layout = QHBoxLayout()

        self.image_widget = ImageWidget()
        self.result_widget = ResultWidget()

        content_layout.addWidget(
            self.image_widget,
            2,
        )

        content_layout.addWidget(
            self.result_widget,
            1,
        )

        upload_button = QPushButton(
            "Upload X-Ray Image"
        )

        upload_button.clicked.connect(
            self.open_image
        )

        analyze_button = QPushButton(
            "Analyze Image"
        )

        analyze_button.clicked.connect(
            self.analyze_image
        )

        # Create horizontal button layout
        button_layout = QHBoxLayout()

        button_layout.addWidget(
            upload_button
        )

        button_layout.addWidget(
            analyze_button
        )

        main_layout.addWidget(title)
        main_layout.addLayout(content_layout)
        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

    def open_image(self) -> None:
        """Open image file."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select X-Ray Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path:
            self.selected_image_path = file_path
            self.image_widget.load_image(file_path)

    def analyze_image(self) -> None:
        """Run AI prediction."""

        if self.selected_image_path is None:
            self.result_widget.update_result(
                "No image selected",
                0,
            )
            return

        prediction, confidence = (
            self.inference_controller.predict(
                self.selected_image_path
            )
        )

        self.result_widget.update_result(
            prediction,
            confidence,
        )


def main() -> None:
    app = QApplication(sys.argv)

    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()