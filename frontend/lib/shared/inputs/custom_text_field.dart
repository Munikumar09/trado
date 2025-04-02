/*
Documentation:
---------------
File: custom_text_field.dart
Description:
  This file defines the CustomTextField widget, a customizable text input field.
  It supports features such as password visibility toggling, read-only mode with a suffix icon action,
  and custom validation. The widget uses a TextFormField and applies custom styling for different input states.

Methods:
  • build(BuildContext context):
      - Constructs the TextFormField with optional suffix icons for password toggle or other actions.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_text_styles.dart';

/// A customizable text input field.
class CustomTextField extends StatefulWidget {
  final String hintText;
  final bool isPassword;
  final TextEditingController? controller;
  final String? labelText;
  final String? Function(String?)? validator;
  final TextInputType? keyboardType;
  final bool readOnly;
  final IconData? suffixIcon;
  final VoidCallback? onSuffixTap;
  final bool autocorrect;
  final bool enableSuggestions;

  const CustomTextField({
    super.key,
    required this.hintText,
    this.isPassword = false,
    this.controller,
    this.labelText,
    this.validator,
    this.keyboardType,
    this.readOnly = false,
    this.suffixIcon,
    this.onSuffixTap,
    this.autocorrect = true,
    this.enableSuggestions = true,
  });

  @override
  State<CustomTextField> createState() => _CustomTextFieldState();
}

class _CustomTextFieldState extends State<CustomTextField> {
  bool _isPasswordVisible = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    Widget? suffix;

    if (widget.isPassword) {
      suffix = IconButton(
        icon: Icon(
          _isPasswordVisible ? Icons.visibility : Icons.visibility_off,
          color: theme.primaryColor,
        ),
        onPressed: () {
          setState(() {
            _isPasswordVisible = !_isPasswordVisible;
          });
        },
      );
    } else if (widget.suffixIcon != null) {
      suffix = IconButton(
        icon: Icon(widget.suffixIcon, color: theme.primaryColor),
        onPressed: widget.onSuffixTap,
      );
    }

    return TextFormField(
      readOnly: widget.readOnly,
      controller: widget.controller,
      obscureText: widget.isPassword && !_isPasswordVisible,
      autocorrect: widget.autocorrect,
      enableSuggestions: widget.enableSuggestions,
      validator: widget.validator ??
          (value) {
            if (value == null || value.isEmpty) {
              final fieldName = widget.labelText ?? 'this field';
              return 'Please enter $fieldName';
            }
            return null;
          },
      autovalidateMode: AutovalidateMode.onUserInteraction,
      keyboardType: widget.keyboardType,
      onTap: widget.readOnly && widget.onSuffixTap != null
          ? widget.onSuffixTap
          : null,
      decoration: InputDecoration(
        labelText: widget.labelText,
        hintText: widget.hintText,
        hintStyle: AppTextStyles.customTextStyle(
            color: theme.hintColor,
            fontSize: bodyText1FontSize,
            fontWeight: bodyText1FontWeight),
        filled: true,
        fillColor: theme.primaryColorLight,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
        enabledBorder: OutlineInputBorder(
          borderSide: BorderSide.none,
          borderRadius: BorderRadius.circular(10),
        ),
        focusedBorder: OutlineInputBorder(
          borderSide: BorderSide(color: theme.primaryColor, width: 2),
          borderRadius: BorderRadius.circular(10),
        ),
        errorBorder: OutlineInputBorder(
          borderSide: BorderSide(color: theme.colorScheme.error),
          borderRadius: BorderRadius.circular(10),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderSide: BorderSide(color: theme.colorScheme.error, width: 2),
          borderRadius: BorderRadius.circular(10),
        ),
        suffixIcon: suffix,
      ),
      style: AppTextStyles.customTextStyle(
          color: theme.primaryColor,
          fontSize: bodyText1FontSize,
          fontWeight: bodyText1FontWeight),
    );
  }
}
