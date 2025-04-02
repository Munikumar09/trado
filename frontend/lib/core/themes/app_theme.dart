/*
Documentation:
---------------
Class: AppThemes
Description:
  Provides light and dark theme configurations for the app using Flutter's ThemeData.
  
Properties:
  • lightTheme:
      - ThemeData for light mode with custom color scheme and styling.
  • darkTheme:
      - ThemeData for dark mode with custom color scheme and styling.

Usage:
  Example: MaterialApp(theme: AppThemes.lightTheme, darkTheme: AppThemes.darkTheme);
*/

// Code:
import 'package:flutter/material.dart';

/// Provides light, dark and custom themes for the app.
class AppThemes {
  /// Theme configuration for light mode.
  static final ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    hintColor: Color(0xFF626262),
    primaryColorLight: Color(0xFFF1F4FF),
    dividerColor: Color(0xFFE5E7EB),
    colorScheme: ColorScheme(
      primary: Color(0xFF1F41BB),
      secondary: Color(0xFF5C53E9),
      surface: Color(0xFFFFFFFF),
      tertiary: Color(0xFF6B7280),
      onTertiary: Color(0xFF000000),
      error: Color(0xFFB00020),
      onPrimary: Color(0xFFFFFFFF),
      onSecondary: Color(0xFF000000),
      onSurface: Color(0xFF000000),
      onError: Color(0xFFFFFFFF),
      brightness: Brightness.light,
    ),
    scaffoldBackgroundColor: Color(0xFFFFFFFF),
    fontFamily: 'Roboto',
  );
}
