import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
)

from PySide6.QtCore import Qt

from PIL import Image

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

        self.setWindowTitle(
            "Medical AI Imaging"
        )

        self.resize(
            1000,
            700,
        )

        self.setMinimumSize(
            900,
            600,
        )

        self.create_ui()

    def create_info_card(self) -> QFrame:
        """Create image information card."""

        card = QFrame()

        card.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border-radius: 15px;
            }
            """
        )

        shadow = QGraphicsDropShadowEffect()

        shadow.setBlurRadius(20)
        shadow.setOffset(0, 5)

        card.setGraphicsEffect(
            shadow
        )

        return card

    def create_ui(self) -> None:
        """Create main user interface."""

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout()

        main_layout.setContentsMargins(
            20,
            20,
            20,
            20,
        )

        main_layout.setSpacing(
            15
        )

        title = QLabel(
            "Medical AI Imaging - Chest X-Ray Analysis"
        )

        title.setObjectName(
            "title"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        content_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.image_widget = ImageWidget()

        self.info_card = self.create_info_card()

        info_layout = QVBoxLayout()

        self.info_title = QLabel(
            "Image Information"
        )

        self.info_title_fa = QLabel(
            "اطلاعات تصویر"
        )

        self.file_name_label = QLabel(
            "File Name: -"
        )

        self.file_name_label_fa = QLabel(
            "نام فایل: -"
        )

        self.size_label = QLabel(
            "Image Size: -"
        )

        self.size_label_fa = QLabel(
            "اندازه تصویر: -"
        )

        for label in [
            self.info_title,
            self.info_title_fa,
            self.file_name_label,
            self.file_name_label_fa,
            self.size_label,
            self.size_label_fa,
        ]:
            label.setAlignment(
                Qt.AlignCenter
            )

        info_layout.addWidget(
            self.info_title
        )

        info_layout.addWidget(
            self.info_title_fa
        )

        info_layout.addWidget(
            self.file_name_label
        )

        info_layout.addWidget(
            self.file_name_label_fa
        )

        info_layout.addWidget(
            self.size_label
        )

        info_layout.addWidget(
            self.size_label_fa
        )

        self.info_card.setLayout(
            info_layout
        )

        left_layout.addWidget(
            self.image_widget
        )

        left_layout.addWidget(
            self.info_card
        )

        self.result_widget = ResultWidget()

        content_layout.addLayout(
            left_layout,
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

        button_layout = QHBoxLayout()

        button_layout.addWidget(
            upload_button
        )

        button_layout.addWidget(
            analyze_button
        )

        main_layout.addWidget(
            title
        )

        main_layout.addLayout(
            content_layout
        )

        main_layout.addLayout(
            button_layout
        )

        central_widget.setLayout(
            main_layout
        )

    def update_image_info(
        self,
        image_path: str,
    ) -> None:
        """Update image information."""

        file_name = Path(
            image_path
        ).name

        with Image.open(
            image_path
        ) as image:

            width, height = image.size

        self.file_name_label.setText(
            f"File Name: {file_name}"
        )

        self.file_name_label_fa.setText(
            f"نام فایل: {file_name}"
        )

        self.size_label.setText(
            f"Image Size: {width} x {height}"
        )

        self.size_label_fa.setText(
            f"اندازه تصویر: {width} x {height}"
        )

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

            self.image_widget.load_image(
                file_path
            )

            self.update_image_info(
                file_path
            )

    def analyze_image(self) -> None:
        """Run AI prediction."""

        if self.selected_image_path is None:

            self.result_widget.show_error(
                "No image selected",
                "تصویری انتخاب نشده است",
            )

            return

        try:

            self.result_widget.show_loading()

            QApplication.processEvents()

            prediction, confidence = (
                self.inference_controller.predict(
                    self.selected_image_path
                )
            )

            self.result_widget.update_result(
                prediction,
                confidence,
            )

        except Exception:

            self.result_widget.show_error(
                "Prediction failed",
                "خطا در انجام تحلیل",
            )


def main() -> None:

    app = QApplication(
        sys.argv
    )

    app.setStyleSheet(
        APP_STYLE
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()