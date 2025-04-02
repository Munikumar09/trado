/*
Documentation:
---------------
File: custom_background_widget.dart
Description:
  This file defines the CustomBackgroundWidget which provides a styled background layout for the app's screens.
  It uses positioned elements and decorative shapes (rectangles and ovals) to create a dynamic background.
  The widget wraps any provided child in a SafeArea and applies custom styling based on screen dimensions.

Methods:
  • build(BuildContext context):
      - Composes the background with decorative positioned widgets and overlays the child widget.
  • _buildRotatedRectangle(...), _buildRectangle(...), _buildOval(...), _buildFilledOval(...):
      - Helper functions to create various decorative shapes.
*/

// Code:
import 'package:flutter/material.dart';

/// A styled background layout for the app's screens.
class CustomBackgroundWidget extends StatelessWidget {
  final Widget? child;

  const CustomBackgroundWidget({super.key, this.child});

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;

    return Container(
      width: double.infinity,
      height: double.infinity,
      clipBehavior: Clip.antiAlias,
      decoration: ShapeDecoration(
        color: Theme.of(context).colorScheme.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(5),
        ),
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: Container(),
          ),
          Positioned(
            left: -0.85 * screenWidth,
            top: -0.7 * screenHeight,
            child: SizedBox(
              width: 2.0 * screenWidth,
              height: 2.0 * screenHeight,
              child: Stack(
                children: [
                  _buildRotatedRectangle(
                    screenWidth,
                    screenHeight,
                    0.4,
                    1.3,
                    0.47,
                    Theme.of(context).primaryColor,
                  ),
                  _buildRectangle(
                    screenWidth,
                    screenHeight,
                    0.14,
                    1.4,
                    Theme.of(context).primaryColor,
                  ),
                  _buildOval(
                    screenWidth,
                    screenHeight,
                    0.95,
                    0.17,
                    2,
                    3,
                    Theme.of(context).primaryColorLight,
                  ),
                  _buildFilledOval(
                    screenWidth,
                    screenHeight,
                    1.12,
                    0.1,
                    2,
                    Theme.of(context).primaryColorLight,
                  ),
                ],
              ),
            ),
          ),
          Positioned(
            left: 0,
            top: 0,
            child: Container(
              width: screenWidth,
              height: 0.05 * screenHeight,
              padding: EdgeInsets.symmetric(
                horizontal: 0.08 * screenWidth,
                vertical: 0.02 * screenHeight,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  SizedBox(
                    width: 0.15 * screenWidth,
                    height: 0.015 * screenHeight,
                  ),
                ],
              ),
            ),
          ),
          if (child != null)
            Positioned.fill(
              child: SafeArea(child: child!),
            ),
        ],
      ),
    );
  }

  Widget _buildRotatedRectangle(
    double screenWidth,
    double screenHeight,
    double left,
    double top,
    double angle,
    Color color,
  ) {
    return Positioned(
      left: left * screenWidth,
      top: top * screenHeight,
      child: Transform(
        transform: Matrix4.identity()..rotateZ(angle),
        child: Container(
          width: 0.87 * screenWidth,
          height: 0.87 * screenWidth,
          decoration: ShapeDecoration(
            shape: RoundedRectangleBorder(
              side: BorderSide(
                width: 2,
                color: color.withValues(alpha: 0.2),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildRectangle(
    double screenWidth,
    double screenHeight,
    double left,
    double top,
    Color color,
  ) {
    return Positioned(
      left: left * screenWidth,
      top: top * screenHeight,
      child: Container(
        width: 0.87 * screenWidth,
        height: 0.87 * screenWidth,
        decoration: ShapeDecoration(
          shape: RoundedRectangleBorder(
            side: BorderSide(
              width: 2,
              color: color.withValues(alpha: 0.2),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildOval(double screenWidth, double screenHeight, double left,
      double top, double size, double borderWidth, Color borderColor) {
    return Positioned(
      left: left * screenWidth,
      top: top * screenHeight,
      child: Container(
        width: size * screenWidth,
        height: size * screenWidth,
        decoration: ShapeDecoration(
          shape: OvalBorder(
            side: BorderSide(width: borderWidth, color: borderColor),
          ),
        ),
      ),
    );
  }

  Widget _buildFilledOval(double screenWidth, double screenHeight, double left,
      double top, double size, Color color) {
    return Positioned(
      left: left * screenWidth,
      top: top * screenHeight,
      child: Container(
        width: size * screenWidth,
        height: size * screenWidth,
        decoration: ShapeDecoration(
          color: color,
          shape: OvalBorder(),
        ),
      ),
    );
  }
}
