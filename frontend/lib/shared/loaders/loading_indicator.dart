/*
Documentation:
---------------
File: loading_indicator.dart
Description:
  This file defines the LoadingIndicator widget which displays a CircularProgressIndicator.
  It provides customizable properties for color, strokeWidth, and size, allowing it to adapt to various UI requirements.

Methods:
  • build(BuildContext context):
      - Builds a SizedBox containing a CircularProgressIndicator with the specified customization.
*/

// Code:
import 'package:flutter/material.dart';

/// A widget that displays a CircularProgressIndicator.
class LoadingIndicator extends StatelessWidget {
  final Color? color;
  final double strokeWidth;
  final double size;

  const LoadingIndicator({
    super.key,
    this.color, // Optional: Customize the color
    this.strokeWidth = 2.0, // Optional: Customize stroke width
    this.size = 20.0, // Optional: Customize the size
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: size,
      width: size,
      child: CircularProgressIndicator(
        strokeWidth: strokeWidth,
        color: color ?? Theme.of(context).colorScheme.onPrimary,
      ),
    );
  }
}
