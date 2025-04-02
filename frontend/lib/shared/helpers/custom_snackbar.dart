/*
Documentation:
---------------
File: custom_snackbar.dart
Description:
  This file provides an extension on BuildContext to show custom snackbars using AwesomeSnackbarContent.
  It defines methods to display snackbars for different scenarios such as error, success, help, and warning messages.

Methods:
  • showCustomSnackBar({required String title, required String message, required ContentType contentType, ...}):
      - Displays a custom snackbar with the provided title, message, and content type.
  • showErrorSnackBar(String message):
      - Displays an error snackbar with a default title.
  • showSuccessSnackBar(String message):
      - Displays a success snackbar with a default title.
  • showHelpSnackBar(String message):
      - Displays a help snackbar with a default title.
  • showWarningSnackBar(String message):
      - Displays a warning snackbar with a default title.
*/

// Code:
import 'package:awesome_snackbar_content/awesome_snackbar_content.dart';
import 'package:flutter/material.dart';

/// Extension on BuildContext to show custom snackbars.
extension SnackbarContext on BuildContext {
  void showCustomSnackBar({
    required String title,
    required String message,
    required ContentType contentType,
    Duration duration = const Duration(seconds: 4),
  }) {
    final snackBar = SnackBar(
      elevation: 0,
      behavior: SnackBarBehavior.floating,
      backgroundColor: Colors.transparent,
      content: AwesomeSnackbarContent(
        title: title,
        message: message,
        contentType: contentType,
      ),
      duration: duration,
    );

    ScaffoldMessenger.of(this)
      ..removeCurrentSnackBar()
      ..showSnackBar(snackBar);
  }

  // Convenience methods (same as before)
  void showErrorSnackBar(String message) {
    showCustomSnackBar(
      title: 'Oh Snap!',
      message: message,
      contentType: ContentType.failure,
    );
  }

  void showSuccessSnackBar(String message) {
    showCustomSnackBar(
      title: 'Success!',
      message: message,
      contentType: ContentType.success,
    );
  }

  void showHelpSnackBar(String message) {
    showCustomSnackBar(
      title: 'Help!',
      message: message,
      contentType: ContentType.help,
    );
  }

  void showWarningSnackBar(String message) {
    showCustomSnackBar(
      title: 'Warning!',
      message: message,
      contentType: ContentType.warning,
    );
  }
}
