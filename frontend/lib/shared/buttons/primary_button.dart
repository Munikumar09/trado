/*
Documentation:
---------------
File: primary_button.dart
Description:
  This file defines the PrimaryButton widget which provides a styled button for primary actions.
  It supports customizable text, colors, font size, border radius, and an optional child widget for displaying loaders.

Methods:
  • build(BuildContext context):
      - Constructs the button using a GestureDetector and Container. If a child is provided, it is displayed instead of text.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_text_styles.dart';

/// A styled button for primary actions.
class PrimaryButton extends StatelessWidget {
  /// The text to display on the button.
  final String text;

  /// The callback function when the button is pressed. If null, the button is disabled.
  final VoidCallback? onPressed;

  /// The background color of the button. Defaults to the theme's primary color.
  final Color? backgroundColor;

  /// The text color of the button. Defaults to the theme's onPrimary color.
  final Color? textColor;

  /// The font size of the button's text. Defaults to 20.
  final double fontSize;

  /// The border radius of the button. Defaults to a circular radius of 10.
  final BorderRadius borderRadius;

  /// An optional child widget to display instead of the text. Useful for
  /// showing loading indicators or other custom content.
  final Widget? child;

  /// {@macro primary_button}
  const PrimaryButton({
    super.key,
    required this.text,
    required this.onPressed,
    this.backgroundColor,
    this.textColor,
    this.fontSize = 20,
    this.borderRadius = const BorderRadius.all(Radius.circular(10)),
    this.child,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return GestureDetector(
      onTap: onPressed, // Use onTap directly, handles null for disabled state
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
        decoration: BoxDecoration(
          color: backgroundColor ?? theme.colorScheme.primary,
          borderRadius: borderRadius,
          boxShadow: [
            BoxShadow(
              color: (backgroundColor ?? theme.colorScheme.primary)
                  .withValues(alpha: 0.3),
              blurRadius: 20,
              offset: const Offset(0, 10),
              spreadRadius: 0,
            ),
          ],
        ),
        child: Center(
          // Use the child if provided, otherwise display the text.
          child: child ??
              Text(
                text,
                style: AppTextStyles.customTextStyle(
                        color: textColor ?? theme.colorScheme.onPrimary,
                        fontSize: heading2FontSize,
                        fontWeight: heading2FontWeight)
                    .copyWith(fontSize: fontSize),
              ),
        ),
      ),
    );
  }
}
