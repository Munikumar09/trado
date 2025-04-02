/*
Documentation:
---------------
File: custom_button.dart
Description:
  This file defines the CustomButton widget which is a customizable button.
  It allows configuration of text, background color, text color, font size, and border radius.
  The widget displays the provided text or an optional child widget for content such as a loading indicator.

Methods:
  • build(BuildContext context):
      - Builds the button using a GestureDetector and Container with custom styling.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_text_styles.dart';

/// A customizable button widget.
class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final Color? backgroundColor;
  final Color? textColor;
  final double fontSize;
  final BorderRadius borderRadius;

  const CustomButton({
    super.key,
    required this.text,
    required this.onPressed,
    this.backgroundColor,
    this.textColor,
    this.fontSize = 20,
    this.borderRadius = const BorderRadius.all(Radius.circular(10)),
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
        decoration: BoxDecoration(
          color: backgroundColor ?? theme.primaryColor,
          borderRadius: borderRadius,
          boxShadow: [
            BoxShadow(
              color: backgroundColor ??
                  theme.primaryColor
                      .withValues(alpha: 0.3), // Shadow matches button color
              blurRadius: 20,
              offset: const Offset(0, 10),
              spreadRadius: 0,
            ),
          ],
        ),
        child: Center(
          child: Text(
            text,
            style: AppTextStyles.customTextStyle(
                color: textColor ?? theme.colorScheme.onPrimary,
                fontSize: heading2FontSize,
                fontWeight: heading2FontWeight),
          ),
        ),
      ),
    );
  }
}
