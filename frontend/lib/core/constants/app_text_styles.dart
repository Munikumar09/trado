/*
Documentation:
---------------
Class: AppTextStyles
Description:
  Provides text styles for various text elements in the app.
  Contains constants for font sizes, font weights, and a method to generate text styles.
*/

// Code:
import 'package:flutter/material.dart';

// Font sizes
const double heading1FontSize = 30;
const double heading2FontSize = 20;
const double heading3FontSize = 18;
const double bodyText1FontSize = 16;
const double bodyText2FontSize = 14;
const double captionFontSize = 12;

// Font weights
const FontWeight heading1FontWeight = FontWeight.w700;
const FontWeight heading2FontWeight = FontWeight.w600;
const FontWeight heading3FontWeight = FontWeight.w600;
const FontWeight bodyText1FontWeight = FontWeight.w500;
const FontWeight bodyText2FontWeight = FontWeight.normal;
const FontWeight captionFontWeight = FontWeight.normal;

/// Provides text styles for various text elements in the app.
class AppTextStyles {
  /// Returns a TextStyle for primary headlines.
  static TextStyle customTextStyle(
      {required color,
      required double fontSize,
      required FontWeight fontWeight,
      double? letterSpacing,
      double? height,
      TextDecoration? decoration}) {
    return TextStyle(
      color: color,
      fontSize: fontSize,
      fontWeight: fontWeight,
      letterSpacing: letterSpacing,
      height: height,
      decoration: decoration,
    );
  }
}
