from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt


class ResultWidget(QWidget):
    """Widget for displaying AI prediction results."""

    LABEL_TRANSLATION = {
        "PNEUMONIA": "پنومونی (ذات‌الریه)",
        "NORMAL": "طبیعی",
    }

    def __init__(self) -> None:
        super().__init__()

        self.create_ui()

    def create_card(self) -> QFrame:
        """Create styled result card."""

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

    def create_label_pair(
        self,
        english_text: str,
        persian_text: str,
    ):
        """Create bilingual label pair."""

        layout = QVBoxLayout()

        english_label = QLabel(
            english_text
        )

        persian_label = QLabel(
            persian_text
        )

        english_label.setAlignment(
            Qt.AlignCenter
        )

        persian_label.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            english_label
        )

        layout.addWidget(
            persian_label
        )

        return layout

    def create_ui(self) -> None:
        """Create result display."""

        main_layout = QVBoxLayout()

        main_layout.setSpacing(20)

        # Prediction Card
        prediction_card = self.create_card()

        prediction_layout = QVBoxLayout()

        self.prediction_title = QLabel(
            "Prediction Result"
        )

        self.prediction_translation = QLabel(
            "نتیجه تشخیص"
        )

        self.prediction_value = QLabel(
            "Waiting..."
        )

        self.prediction_value_fa = QLabel(
            "در انتظار..."
        )

        for label in [
            self.prediction_title,
            self.prediction_translation,
            self.prediction_value,
            self.prediction_value_fa,
        ]:
            label.setAlignment(
                Qt.AlignCenter
            )

        prediction_layout.addWidget(
            self.prediction_title
        )

        prediction_layout.addWidget(
            self.prediction_translation
        )

        prediction_layout.addWidget(
            self.prediction_value
        )

        prediction_layout.addWidget(
            self.prediction_value_fa
        )

        prediction_card.setLayout(
            prediction_layout
        )

        # Confidence Card
        confidence_card = self.create_card()

        confidence_layout = QVBoxLayout()

        self.confidence_title = QLabel(
            "Confidence"
        )

        self.confidence_translation = QLabel(
            "میزان اطمینان"
        )

        self.confidence_bar = QProgressBar()

        self.confidence_bar.setRange(
            0,
            100,
        )

        self.confidence_bar.setValue(
            0
        )

        self.confidence_bar.setMinimumHeight(
            22
        )

        self.confidence_bar.setAlignment(
            Qt.AlignCenter
        )

        self.confidence_bar.setTextVisible(
            True
        )

        for label in [
            self.confidence_title,
            self.confidence_translation,
        ]:
            label.setAlignment(
                Qt.AlignCenter
            )

        confidence_layout.addWidget(
            self.confidence_title
        )

        confidence_layout.addWidget(
            self.confidence_translation
        )

        confidence_layout.addWidget(
            self.confidence_bar
        )

        confidence_card.setLayout(
            confidence_layout
        )

        # Status Card
        status_card = self.create_card()

        status_layout = QVBoxLayout()

        self.status_title = QLabel(
            "Status"
        )

        self.status_translation = QLabel(
            "وضعیت"
        )

        self.status_value = QLabel(
            "No analysis"
        )

        self.status_value_fa = QLabel(
            "بدون تحلیل"
        )

        for label in [
            self.status_title,
            self.status_translation,
            self.status_value,
            self.status_value_fa,
        ]:
            label.setAlignment(
                Qt.AlignCenter
            )

        status_layout.addWidget(
            self.status_title
        )

        status_layout.addWidget(
            self.status_translation
        )

        status_layout.addWidget(
            self.status_value
        )

        status_layout.addWidget(
            self.status_value_fa
        )

        status_card.setLayout(
            status_layout
        )

        main_layout.addWidget(
            prediction_card
        )

        main_layout.addWidget(
            confidence_card
        )

        main_layout.addWidget(
            status_card
        )

        self.setLayout(
            main_layout
        )

    def update_result(
        self,
        prediction: str,
        confidence: float,
    ) -> None:
        """Update prediction result."""

        translated_prediction = self.LABEL_TRANSLATION.get(
            prediction,
            "نامشخص",
        )

        self.prediction_value.setText(
            prediction
        )

        self.prediction_value_fa.setText(
            translated_prediction
        )

        self.confidence_bar.setValue(
            int(confidence)
        )

        self.confidence_bar.setFormat(
            f"{confidence:.2f}%"
        )

        # Change progress bar color based on confidence level
        if confidence < 50:
            color = "#ef4444"
        elif confidence < 80:
            color = "#f97316"
        else:
            color = "#22c55e"

        self.confidence_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                background-color: #f8fafc;
                text-align: center;
                font-weight: bold;
            }}

            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 8px;
            }}
            """
        )

        self.status_value.setText(
            "Analysis completed"
        )

        self.status_value_fa.setText(
            "تحلیل کامل شد"
        )