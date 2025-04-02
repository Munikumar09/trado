/*
Documentation:
---------------
File: icon_button.dart
Description:
  This file implements the CustomIconButton widget which wraps an image asset inside an IconButton.
  It provides error handling for the image loading and is intended for use as a social login button or similar actions.

Methods:
  • build(BuildContext context):
      - Returns an IconButton with the image asset as its icon and error handling.
*/

// Code:
import 'package:flutter/material.dart';

/// A widget that displays an image asset inside an IconButton.
class CustomIconButton extends StatelessWidget {
  final VoidCallback onPressed;
  final String iconPath;

  const CustomIconButton(
      {super.key, required this.onPressed, required this.iconPath});

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: onPressed,
      icon: Image.asset(
        iconPath,
        width: 27,
        height: 27,
        errorBuilder: (context, error, stackTrace) {
          return const Icon(Icons.error_outline);
        },
      ),
    );
  }
}
