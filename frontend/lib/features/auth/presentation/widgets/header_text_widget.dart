/*
Documentation:
---------------
File: header_text_widget.dart
Description:
  This file implements the HeaderTextWidget which displays a header title and an optional subtitle.
  It allows customization for title and subtitle text colors and provides semantic labels for accessibility.

Methods:
  • build(BuildContext context):
      - Constructs the widget layout displaying the title and, if provided, the subtitle.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_text_styles.dart';

/// A widget that displays a header title and an optional subtitle.
class HeaderTextWidget extends StatelessWidget {
  const HeaderTextWidget({
    super.key,
    required this.title,
    this.subtitle,
    this.titleColor,
    this.subtitleColor,
  });

  final String title;
  final String? subtitle;
  final Color? titleColor;
  final Color? subtitleColor;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          title,
          style: AppTextStyles.customTextStyle(
            color: titleColor ?? theme.primaryColor,
            fontSize: heading1FontSize,
            fontWeight: heading1FontWeight,
          ),
          // Add semantic properties for accessibility
          semanticsLabel: title,
        ),
        const SizedBox(height: 16),
        if (subtitle != null)
          Align(
            alignment: Alignment.center,
            child: Text(
              subtitle!,
              style: AppTextStyles.customTextStyle(
                color: subtitleColor ?? theme.colorScheme.onSurface,
                fontSize: heading2FontSize,
                fontWeight: heading2FontWeight,
              ),
              semanticsLabel: subtitle,
            ),
          ),
      ],
    );
  }
}
