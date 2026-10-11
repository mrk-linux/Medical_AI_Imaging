from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
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

        card.setGraphicsEffect(shadow)

        return card

    def create_indicator(self) -> QLabel:
        """Create status indicator."""

        indicator = QLabel()

        indicator.setFixedSize(
            18,
            18,
        )

        indicator.setStyleSheet(
            """
            QLabel {
                background-color: #22c55e;
                border-radius: 9px;
            }
            """
        )

        return indicator

    def create_ui(self) -> None:
        """Create result display."""

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)

        prediction_card = self.create_card()

        prediction_layout = QVBoxLayout()

        title_layout = QHBoxLayout()

        self.prediction_indicator = self.create_indicator()

        self.prediction_title = QLabel(
            "Prediction Result"
        )

        title_layout.addWidget(
            self.prediction_indicator
        )

        title_layout.addWidget(
            self.prediction_title
        )

        title_layout.setAlignment(
            Qt.AlignCenter
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
            label.setAlignment(Qt.AlignCenter)

        prediction_layout.addLayout(
            title_layout
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

        self.confidence_bar.setValue(0)

        self.confidence_bar.setMinimumHeight(
            22
        )

        self.confidence_bar.setAlignment(
            Qt.AlignCenter
        )

        self.confidence_bar.setTextVisible(
            True
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

    def update_indicator(
        self,
        color: str,
    ) -> None:
        """Update indicator color."""

        self.prediction_indicator.setStyleSheet(
            f"""
            QLabel {{
                background-color: {color};
                border-radius: 9px;
            }}
            """
        )

    def show_loading(self) -> None:
        """Show analysis loading state."""

        self.update_indicator(
            "#facc15"
        )

        self.prediction_title.setText(
            "Analyzing..."
        )

        self.prediction_translation.setText(
            "در حال تحلیل تصویر"
        )

        self.prediction_value.setText(
            "Please wait"
        )

        self.prediction_value_fa.setText(
            "لطفاً منتظر بمانید"
        )

    def show_error(
        self,
        message_en: str,
        message_fa: str,
    ) -> None:
        """Display error message."""

        self.update_indicator(
            "#ef4444"
        )

        self.prediction_title.setText(
            "Error"
        )

        self.prediction_translation.setText(
            "خطا"
        )

        self.prediction_value.setText(
            message_en
        )

        self.prediction_value_fa.setText(
            message_fa
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

        self.prediction_title.setText(
            "Prediction Result"
        )

        self.prediction_translation.setText(
            "نتیجه تشخیص"
        )

        self.prediction_value.setText(
            prediction
        )

        self.prediction_value_fa.setText(
            translated_prediction
        )

        if prediction == "NORMAL":
            self.update_indicator("#22c55e")

        elif prediction == "PNEUMONIA":
            self.update_indicator("#f97316")

        else:
            self.update_indicator("#ef4444")

        self.confidence_bar.setValue(
            int(confidence)
        )

        self.confidence_bar.setFormat(
            f"{confidence:.2f}%"
        )

        self.status_value.setText(
            "Analysis completed"
        )

        self.status_value_fa.setText(
            "تحلیل کامل شد"
        )