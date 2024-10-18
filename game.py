# coding:utf-8
import os
import sys
from PyQt5.QtCore import Qt, QTranslator, QLocale
from PyQt5.QtWidgets import QApplication
from app.View.main_window import MainWindow
from app.common.os_utils import getDevicePixelRatio


# Enable high-DPI scaling and adjust scaling factor for better display
def configure_high_dpi_scaling():
    """
    Configures high-DPI scaling settings for the application.
    Adjusts the scale factor based on the device's pixel ratio.
    """
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR"] = str(max(1, getDevicePixelRatio() - 0.25))
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)


def configure_application_attributes():
    """
    Sets up application-specific attributes to improve rendering.
    Avoids native widget siblings for better UI performance.
    """
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)


def setup_translator(app):
    """
    Sets up the application translator for internationalization.
    Loads language settings based on the system locale.

    Parameters
    ----------
    app : QApplication
        The main application instance for which the translator is being set.
    """
    translator = QTranslator()
    translator.load(QLocale.system(), ':/i18n/AlphaGobangZero_')
    app.installTranslator(translator)


def create_main_window():
    """
    Creates and displays the main application window.

    Returns
    -------
    MainWindow
        The main window instance.
    """
    main_window = MainWindow(board_len=9)
    main_window.show()
    return main_window


def main():
    """
    Main function to set up and run the AlphaGobangZero application.
    Configures high-DPI scaling, application attributes, and translation,
    then creates and displays the main game window.
    """
    configure_high_dpi_scaling()

    # Create the main application instance
    app = QApplication(sys.argv)

    configure_application_attributes()
    setup_translator(app)

    # Create and show the main game window
    main_window = create_main_window()

    # Start the application event loop
    sys.exit(app.exec_())


# Entry point for the application
if __name__ == "__main__":
    main()
